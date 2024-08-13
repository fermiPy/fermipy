.. _team:

Project & Team
##############

This page describes the *fermipy* project organisation and the main roles and responsibilities in the *fermipy* team.
This structure was set in place in July 2024, but we expect this structure to evolve over the coming years,
adapting to the size and composition of the *fermipy* development team, and the requirements and needs of scientists and
projects using *fermipy*.
If you would like to become part of the *fermipy* team, please get in contact. **Help is always welcome!**


Overview
********
The following sections describe the major roles and responsibilities in the *fermipy* team:

* `Coordination Committee`_
* `Principal Investigator`_
* `Lead developers`_
* `Sub-package maintainers`_
* `Contributors and previous core developers`_
* `Supporting institutions`_
* `Grants`_


Coordination Committee
************************

The *fermipy* coordination committee (CC) is the board that is responsible to promote, coordinate and steer *fermipy* developments.
It also serves as main contact point for the *fermipy* project.

*fermipy* is developed mostly by member of the Fermi-LAT collaboration, and it is used by people and projects from several
institutes and countries.
The CC is composed by the analysis coordinator (and their deputy) of the Fermi LAT collaboration, the *fermipy* principal investigator
and lead developers.
The CC also includes representation from the Fermi Science Support Center (`FSSC <https://fermi.gsfc.nasa.gov/ssc/>`_) and Fermi User Group (`FUG <https://fermi.gsfc.nasa.gov/ssc/library/fug/>`_).


Responsibilities include:
=========================

- Being the point of contact for the *fermipy* project.
- Promote the use of *fermipy* by new projects, especially as science tool for Fermi-LAT.
- Keep the overview of ongoing activities, schedules and action items and follow up to make sure all important things get done.
- Make decisions on the scope, content and development priorities for the *fermipy* package.
- Support and grow the *fermipy* team (help find manpower and funding)
- Support and coordinate the use of *fermipy* for scientific or technical studies and papers
- Organise and drive all non-technical aspects of the project on a day-to-day basis.
- Keep an overview and help coordinate all activities that have some involvement of *fermipy*, such as e.g. papers, presentations or posters about or using *fermipy* at gamma-ray astronomy meetings or conferences, or tutorials at schools / workshops on gamma-ray astronomy data analysis.
- Manage the *fermipy* developer / maintainer / contributor team. Distribute tasks and assign responsibilities to other *fermipy* developers.
- Ensure that anyone interested in contributing to *fermipy* development has good resources (documentation, communication, mentoring) to get started. Specifically: maintain the *fermipy* developer documentation that describes all aspects of *fermipy* development (code, testing, documentation, processes).
- Organise *fermipy* developer calls and coding sprints via *fermipy*-meetings
- Schedule *fermipy* releases and define which fixes and features go in which release, taking the needs of people and projects using *fermipy* as well as available manpower for developments into account. Either execute releases directly or assign a release manager.
- Monitor and assign of issues and pull requests,
- Ensure *fermipy* infrastructure is well set up and maintained (issue tracker and pull requests on Github, continuous integration tests, documentation builds, releases and distribution).

**Current CC members (alphabetical order):**

* Niccolò Di Lalla (Core Developer) - Stanford University
* Leonardo Di Venere (Core Developer) - INFN Bari
* Michela Negro (Fermi-LAT Analysis Deputy Coordinator) - Louisiana State University
* Nicola Omodei  (PI, Core Developer) - Stanford University
* Giacomo Principe (Core Developer) - INFN Trieste
* Miguel Sánchez-Conde (Fermi-LAT Analysis Coordinator) - Universidad Autónoma de Madrid
* Marcos Santander (FUG) - University of Alabama
* Alex Reustle (FSSC) - NASA GSFC


Principal Investigator
************************

The *fermipy* Principal Investigator (PI) is in charge of seeking funding,
they overview the work and work closely with the *fermipy* coordination committee, lead developers, contributors and users.

Responsibilities include:
=========================
- Maintain *fermipy* communication channels (mailing lists, slack, github, ...)
- Serve as *fermipy* coordination committee secretary (schedule and moderate calls; give status reports; write minutes)
- Serve on the *fermipy* coordination committee, as the link between CC and the development team.
- Appoint the *fermipy*  managers (non-technical lead) and lead developers (technical lead)
- Organise *fermipy* user calls and training events via *fermipy*-meetings
- Review the documents are properly reviewed and eventually decisions made by the CC.

**Current fermipy PI:**

* Nicola Omodei  (PI, Core Developer) - Stanford University

Lead developers
*****************
The lead developers are the technical executive leads for the *fermipy* project.
The lead developers are appointed by the *fermipy* coordination committee,
and work closely with the *fermipy* coordination committee, project managers and contributors.

Responsibilities include:
=========================

- Organize and drive all technical aspects of the project on a day-to-day basis. Keep the overview of ongoing activities, schedules and action items and follow up to make sure all important things get done.
- Evaluating new pull requests for quality, API consistency and *fermipy* coding standards,
- Supporting developers on tasks associated to the sub-package(s),
- Taking care of the global design of the sub-package(s) in the context of the global *fermipy* architecture, participating to the User Support for questions related to the sub-package(s).
- Solve, comment, or re-assign issues and pull requests.

**Current fermipy lead developers:**

* Niccolò Di Lalla (Core Developer) - Stanford University
* Leonardo Di Venere (Core Developer) - INFN Bari
* Nicola Omodei  (PI, Core Developer) - Stanford University
* Giacomo Principe (Core Developer) - INFN Trieste

Sub-package maintainers
**********************************

Among the *fermipy* core developer team, they are some experts that are devoted to the maintenance of some sub-packages.

Responsibilities include:
=========================
- Solve, comment or reassign issues and pull requests.
- support development on tasks associated to the sub-package(s),
- evaluating new pull requests for quality, API consistency and *fermipy* coding standards,
- taking care of the global design of the sub-package(s) in the context of the global *fermipy* architecture,
- participating to the User Support for questions related to the sub-package(s).

**List of sub-package (with assigned maintainers):**

* Catalogs (data, format) -
* Diffuse (GalProp, MapCube) - Troy Porter - Stanford University
* SED -
* Localization - Niccolò Di Lalla - Stanford University
* Lightcurve - Janeth Valverde - NASA GSFC
* Jobs (managing pipelines) - Nicola Omodei, Niccolò Di Lalla - Stanford University

Contributors and previous core developers
***********************************************
Some of the original *fermipy* developer have left the academia or move to different jobs.
Nonetheless we want to acknowledge their original involvement and vision in creating *fermipy*.

* Matt Wood
* Eric Charles
* Henrike Fleischhack
* Mattia Di Mauro
* Sara Buson
* Anna Franckowiak
* Alex Drlica-Wagner
* Rolf Buehler
* Terri Brandt
* Joe Acercion
* Stephan Zimmer
* James Chiang
* Andy Smith

`List of all contributors <https://github.com/fermiPy/fermipy/graphs/contributors>`_


Supporting institutions
****************************

People involved in *fermipy* are coming from different institutions, laboratories and universities.
We acknowledge them for their daily support.


Grants
********
Grants that are supporting the development of *fermipy*:

* Fermi GI cycle 16 (Large project): proposal n. 161029. P.I.: Nicola Omodei (Stanford University)

