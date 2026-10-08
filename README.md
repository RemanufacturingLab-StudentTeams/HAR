# Remanufacturing Lab - "Project Name"
"Brief explanation of project"

# General Overview
## File structure
This project consists of several (mostly) heterogeneous components, that are kept in separate project-level directories with their own documentation. These are:
- "Give a brief explanation of every component in the repository."

## Testing

(Automated) testing is incredibly important, both for ensuring code correctness and to formally lay down expected behaviour. The different components have different testing requirements:

### Version control
A [Git Policy](docs/policy.md) was written for this project. Please adhere to it as tightly as possible during development.

### Documentation

Documentation requirements are different depending on the subject:
- Hardware: Please include a copy of any hardware manuals within the `docs` directory. 
- Non-proprietary software: A `README.md` file suffices. This file should explain why this software is used, how to set it up locally, and how to deploy it in production.
- Proprietary software and config files: A `README.md file` is necessary, and comments should be written within the code itself explaining what it is for.
- Schemas: These should be extensively documented. A comprehensive overview of the schema should be in the `schemas` directory, with a detailed `README.md`. The idea is that a collaborator should not have to be able to understand the internal workings of a component in order to still communicate with it over a given network protocol.


# Attribution

- Project manager: [Manager of the project/project leader])
- Collaborator - "roles of the collaborator": [Name]
