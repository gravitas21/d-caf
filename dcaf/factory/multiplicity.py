"""
This module should help introducing multiplicity into an existent star list.

The default method idea is to keep the distribution mostly fixed but group stars
together around a primary. No change to the IMF and only change the positions
and velocity of the companions.
Then, we proceed as follows:

1. we choose which stars become primordial binaries
2. pick companions by adjusting to a predefined mass ratio distribution
3. calculate the orbits of the systems based on binary populations, probably a
user defined function could also work.
4. Put all stars together at the location of the primary, using its coordinates
as center of mass.

"""

import numpy

from amuse.lab import Particles, units
from amuse.units.constants import G


class BinaryPopulation:
    """
    Thin class wrapper around the functional binary-building workflow.

    This keeps the current helper functions available while offering a single
    configured object that can be subclassed later for custom population
    choices.
    """

    def __init__(
        self,
        nbinaries=None,
        population_fraction=None,
        gamma=0.0,
        q_min=0.0,
        mean_period=10.0 ** 4.8 | units.day,
        sigma_logP=2.3,
        eccentricities="circular",
        max_radius=None,
    ):
        self.nbinaries = nbinaries
        self.population_fraction = population_fraction
        self.gamma = gamma
        self.q_min = q_min
        self.mean_period = mean_period
        self.sigma_logP = sigma_logP
        self.eccentricities = eccentricities
        self.max_radius = max_radius

        if (self.nbinaries is None) == (self.population_fraction is None):
            raise ValueError("Provide exactly one of `nbinaries` or `population_fraction`.")
        if self.population_fraction is not None:
            if len(self.population_fraction) < 2:
                raise ValueError(
                    "population_fraction must contain at least [single_fraction, binary_fraction]."
                )
            if numpy.any(numpy.array(self.population_fraction) < 0.0):
                raise ValueError("population_fraction entries must be non-negative.")
            if numpy.sum(self.population_fraction) <= 0.0:
                raise ValueError("population_fraction must have a positive sum.")
            if len(self.population_fraction) > 2:
                extra = numpy.array(self.population_fraction[2:])
                if numpy.any(extra > 0.0):
                    raise NotImplementedError(
                        "BinaryPopulation currently supports only singles and binaries."
                    )
        if self.q_min < 0.0 or self.q_min > 1.0:
            raise ValueError("q_min must be between 0 and 1.")

    def get_number_of_binaries(self, stars):
        """
        Determine how many binary systems to construct from the configured
        population parameters.
        """
        if self.nbinaries is not None:
            nbinaries = self.nbinaries
        else:
            fractions = numpy.array(self.population_fraction, dtype=float)
            fractions = fractions / fractions.sum()
            mean_multiplicity = numpy.sum(
                fractions * numpy.arange(1, len(fractions) + 1, dtype=float)
            )
            nsystems = float(len(stars)) / mean_multiplicity
            nbinaries = int(numpy.floor(fractions[1] * nsystems))

        if nbinaries < 0:
            raise ValueError("nbinaries must be non-negative.")
        if nbinaries > len(stars) // 2:
            raise ValueError("nbinaries can not exceed len(stars)//2.")

        return nbinaries

    def choose_companion(self, m1, mass, equal_mass=False):
        """
        Select one companion mass from the available pool using the configured
        mass-ratio distribution p(q) ∝ q^gamma, where q = m2/m1 <= 1.
        """
        if len(mass) == 0:
            raise ValueError("No companion masses available.")

        if equal_mass:
            index = numpy.random.randint(len(mass))
            return mass[index]

        if m1 <= mass.min():
            raise Exception("m1 cannot be the smaller mass in the set")

        q = mass.value_in(m1.unit) / m1.value_in(m1.unit)
        valid = (q <= 1.0) & (q >= self.q_min)

        if not numpy.any(valid):
            raise Exception("No available companion masses satisfy q_min <= q <= 1.")

        valid_index = numpy.where(valid)[0]
        q_valid = q[valid]
        weights = q_valid ** self.gamma

        if not numpy.all(numpy.isfinite(weights)):
            raise ValueError("Non-finite weights encountered in mass-ratio sampling.")
        if weights.sum() <= 0:
            raise ValueError("Mass-ratio weights sum to zero.")

        weights = weights / weights.sum()
        choice = numpy.random.choice(valid_index, p=weights)
        return mass[choice]

    def get_eccentricities(self, nbinaries):
        """
        Sample binary eccentricities according to the configured eccentricity
        population model.
        """
        if self.eccentricities == "thermal":
            return numpy.sqrt(numpy.random.uniform(size=nbinaries))
        if self.eccentricities == "circular":
            return numpy.zeros(nbinaries)
        if self.eccentricities == "flat":
            return numpy.random.uniform(size=nbinaries)
        raise Exception(
            "Eccentricity population '%s' not implemented" % self.eccentricities
        )

    def select_binaries(self, mass, nbinaries):
        equal_mass = numpy.std(mass.value_in(units.MSun)) == 0.0

        primary_index = []
        companion_index = []
        available_indexes = list(range(len(mass)))
        mmin = mass.min()

        for _ in range(nbinaries):
            if len(available_indexes) > 2:
                mindex = mmin
                if not equal_mass:
                    while mindex == mmin:
                        index = numpy.random.choice(available_indexes)
                        mindex = mass[index]
                        if len(available_indexes) == 1:
                            break
                else:
                    index = numpy.random.choice(available_indexes)
                    mindex = mass[index]

                primary_index.append(index)
                available_indexes.remove(index)

                m2 = self.choose_companion(mass[index], mass[available_indexes], equal_mass)
                index2 = index
                while index2 not in available_indexes:
                    candidates = numpy.where(mass == m2)[0]
                    index2 = numpy.random.choice(candidates)
                    mindex = mass[index2]

                companion_index.append(index2)
                available_indexes.remove(index2)

                if mindex == mmin and len(available_indexes) > 0:
                    mmin = mass[available_indexes].min()
            else:
                if len(available_indexes) == 2:
                    index1 = available_indexes[0]
                    index2 = available_indexes[1]
                    if mass[index1] >= mass[index2]:
                        primary_index.append(index1)
                        companion_index.append(index2)
                    else:
                        primary_index.append(index2)
                        companion_index.append(index1)
                break

        return primary_index, companion_index

    def make_binaries(
        self,
        stars,
    ):
        """
        Resolve binary components around the centre-of-mass phase-space points
        of the selected primary stars, using the current class configuration.
        """
        mass = stars.mass
        nbinaries = self.get_number_of_binaries(stars)
        primary_index, companion_index = self.select_binaries(mass, nbinaries)
        binary_index = list(primary_index)
        used = set(primary_index) | set(companion_index)
        single_index = [i for i in range(len(stars)) if i not in used]

        Rc = self.max_radius
        if Rc is None:
            Rc = stars.LagrangianRadii(mf=[0.5], cm=stars.center_of_mass())[0][0]

        nbinaries = len(primary_index)

        if nbinaries == 0:
            resolved_stars = stars.copy()
            resolved_stars.system_id = numpy.arange(1, len(resolved_stars) + 1, dtype=int)
            return (
                {
                    "resolved_stars": resolved_stars,
                    "unresolved_stars": resolved_stars.copy(),
                    "periods": [] | units.day,
                    "semi_major_axes": [] | units.AU,
                    "eccentricities": numpy.array([]),
                    "nbinaries": 0,
                },
                {
                    "primary_components": Particles(),
                    "secondary_components": Particles(),
                    "single_particles": resolved_stars.copy(),
                    "binary_systems": Particles(),
                    "primary_index": [],
                    "companion_index": [],
                    "single_index": list(range(len(stars))),
                },
            )

        primary_index = list(primary_index)
        companion_index = list(companion_index)
        binary_index = list(binary_index)
        single_index = list(single_index)

        binary_mass = mass[primary_index] + mass[companion_index]
        Pmax = ((Rc.value_in(units.AU) ** 3) / 2.0 / 0.01) ** 0.5 | units.day
        Pmin = 1.0 | units.day

        periods = numpy.zeros(nbinaries) | units.day
        pending = list(range(nbinaries))
        while len(pending) > 0:
            logP = numpy.random.normal(
                loc=numpy.log10(self.mean_period.value_in(units.day)),
                scale=self.sigma_logP,
                size=len(pending),
            )
            periods[pending] = 10.0 ** logP | units.day
            pending = list(numpy.where((periods > Pmax) | (periods < Pmin))[0])

        ecc = self.get_eccentricities(nbinaries)
        semi_major_axes = (
            (periods.value_in(units.yr) ** 2) * binary_mass.value_in(units.MSun)
        ) ** (1.0 / 3.0) | units.AU

        vc = (G * binary_mass * (1.0 - ecc) / semi_major_axes / (1.0 + ecc)).sqrt()
        separation = semi_major_axes * (1.0 + ecc)

        pi_angle = numpy.random.uniform(high=2.0 * numpy.pi, size=nbinaries)
        omega = numpy.random.uniform(high=2.0 * numpy.pi, size=nbinaries)
        zi = numpy.random.uniform(high=numpy.pi, size=nbinaries)

        px1 = numpy.cos(pi_angle) * numpy.cos(omega) - numpy.sin(pi_angle) * numpy.sin(omega) * numpy.cos(zi)
        qx1 = -numpy.sin(pi_angle) * numpy.cos(omega) - numpy.cos(pi_angle) * numpy.sin(omega) * numpy.cos(zi)
        px2 = numpy.cos(pi_angle) * numpy.sin(omega) + numpy.sin(pi_angle) * numpy.cos(omega) * numpy.cos(zi)
        qx2 = -numpy.sin(pi_angle) * numpy.sin(omega) + numpy.cos(pi_angle) * numpy.cos(omega) * numpy.cos(zi)
        px3 = numpy.sin(pi_angle) * numpy.sin(zi)
        qx3 = numpy.cos(pi_angle) * numpy.sin(zi)

        xrel = px1 * separation
        yrel = px2 * separation
        zrel = px3 * separation
        vxrel = qx1 * vc
        vyrel = qx2 * vc
        vzrel = qx3 * vc

        binary_systems = Particles(nbinaries)
        binary_systems.mass = binary_mass
        binary_systems.position = stars[binary_index].position
        binary_systems.velocity = stars[binary_index].velocity
        binary_systems.m1 = mass[primary_index]
        binary_systems.m2 = mass[companion_index]
        binary_systems.binary = True

        primary_components = Particles(nbinaries)
        secondary_components = Particles(nbinaries)

        primary_components.mass = mass[primary_index]
        secondary_components.mass = mass[companion_index]
        primary_components.radius = stars[binary_index].radius
        secondary_components.radius = stars[binary_index].radius

        primary_components.x = binary_systems.x + secondary_components.mass * xrel / binary_systems.mass
        primary_components.y = binary_systems.y + secondary_components.mass * yrel / binary_systems.mass
        primary_components.z = binary_systems.z + secondary_components.mass * zrel / binary_systems.mass

        secondary_components.x = primary_components.x - xrel
        secondary_components.y = primary_components.y - yrel
        secondary_components.z = primary_components.z - zrel

        primary_components.vx = binary_systems.vx + secondary_components.mass * vxrel / binary_systems.mass
        primary_components.vy = binary_systems.vy + secondary_components.mass * vyrel / binary_systems.mass
        primary_components.vz = binary_systems.vz + secondary_components.mass * vzrel / binary_systems.mass

        secondary_components.vx = primary_components.vx - vxrel
        secondary_components.vy = primary_components.vy - vyrel
        secondary_components.vz = primary_components.vz - vzrel

        primary_components.in_binary = True
        secondary_components.in_binary = True

        single_particles = stars[single_index].copy()
        if len(single_particles) > 0:
            single_particles.binary = False
            single_particles.in_binary = False

        return (
            {
                "periods": periods,
                "semi_major_axes": semi_major_axes,
                "eccentricities": ecc,
                "nbinaries": nbinaries,
            },
            {
                "primary_components": primary_components,
                "secondary_components": secondary_components,
                "single_particles": single_particles,
                "binary_systems": binary_systems,
                "primary_index": primary_index,
                "companion_index": companion_index,
                "single_index": single_index,
            },
        )

    def apply(self, stars):
        """
        Build the configured primordial-binary population on top of an existing
        stellar catalog and return both the resolved stars and the unresolved
        system-level particle set.
        """
        data, internal = self.make_binaries(stars)

        nbinaries = data["nbinaries"]

        if nbinaries > 0:
            ordered_systems = []
            for i in range(nbinaries):
                ordered_systems.append(
                    (
                        min(internal["primary_index"][i], internal["companion_index"][i]),
                        "binary",
                        i,
                    )
                )
            for i, idx in enumerate(internal["single_index"]):
                ordered_systems.append((idx, "single", i))
            ordered_systems.sort(key=lambda item: item[0])

            resolved_stars = Particles()
            unresolved_stars = Particles()
            system_id = 1

            for _, kind, i in ordered_systems:
                if kind == "binary":
                    primary = internal["primary_components"][i].copy()
                    secondary = internal["secondary_components"][i].copy()
                    binary_system = internal["binary_systems"][i].copy()

                    primary.system_id = system_id
                    secondary.system_id = system_id
                    binary_system.system_id = system_id

                    resolved_stars.add_particle(primary)
                    resolved_stars.add_particle(secondary)
                    unresolved_stars.add_particle(binary_system)
                else:
                    single = internal["single_particles"][i].copy()
                    single.system_id = system_id

                    resolved_stars.add_particle(single)
                    unresolved_stars.add_particle(single.copy())

                system_id += 1

            data["resolved_stars"] = resolved_stars
            data["unresolved_stars"] = unresolved_stars

        unresolved_stars = data["unresolved_stars"]

        if self.population_fraction is None:
            population_fraction = [
                float(len(unresolved_stars) - nbinaries) / len(unresolved_stars)
                if len(unresolved_stars) > 0
                else 0.0,
                float(nbinaries) / len(unresolved_stars)
                if len(unresolved_stars) > 0
                else 0.0,
            ]
        else:
            population_fraction = [0.0] * max(2, len(self.population_fraction))
            if len(unresolved_stars) > 0:
                population_fraction[0] = (
                    float(len(unresolved_stars) - nbinaries) / len(unresolved_stars)
                )
                population_fraction[1] = (
                    float(nbinaries) / len(unresolved_stars)
                )

        data["nbinaries"] = nbinaries
        data["population_fraction"] = population_fraction
        data["binary_fraction"] = population_fraction[1]

        return data
