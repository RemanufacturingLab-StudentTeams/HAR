# Remanufacturing Lab Git Policy 

This document details the Git policy for software development for the GitHub repository for the THUAS Remanufacturing Lab. All contributors to the project that want to introduce mutations (including small fixes) intended to persist to the codebase should follow the policy outlined here. If your aim is to fork the code and experiment with it without introducing persistent changes, please just clone the repo and do not mutate it.

The GitHub repository is set to private and can only be accessed by authorized users.

## Documentation
There is a README.md file at the project level, which contains general architecture information about the project and lists all the individual components. All contributors should also put their Git username and their real name next to it, so there is a possibility to contact people who introduce changes.

### Components

The project, at the time of writing, consists of several mostly heterogenous software components. They are kept in separate directories at the project level. 

The components have different requirements to run, and some have hardware requirements that make them impossible to test locally. They will mostly be developed independently, and have separate README files and documentation.

It is important to note that even code that cannot be ran/tested locally because of hardware requirements or is not even human-readable without interpreter software should *still* be added to the GitHub repository, so mutations can be tracked and documented properly.

Whenever possible, such software should be stubbed as a hardware-independent program. More on that [here](#testing).

### Network documentation and data schemas

Between the components, there are certain network protocols (HTTP, TCP, MQTT, OPC/UA) that make communication possible. These should be documented as extensively as possible. At the minimum, all the information that is required to consume and send data to another component should be documented, so components can be developed without the dev needing to understand how other components work internally. 

Mandatory things to document per protocol:
- MQTT
- - Broker address and port
- - Every topic
- - (per topic) JSON schema
- - (per topic) description, i.d., what is the point of the data sent on this topic
- OPC/UA
- - Sever address and port
- - Complete Information Model/Namespace
- - Every endpoint
- - (per endpoint) description 
- - (possibly) Security configuration (certificates, credentials etc)
- HTTP/REST API
- - Server address and port
- - DTO schemas in JSON format
- - Every endpoint
- - (per endpoint) description
- - (possibly) Security related configurations (API key, CSRF token, CORS policy etc)
- (list may be expanded in the future)

If any custom configuration was done for an endpoint, be it authorization, flow control, error feedback, or whatever, it needs to be documented.

## Commits

Commits should be atomic, and preferably small. [This convention](https://www.conventionalcommits.org/en/v1.0.0/) should be used for writing commit messages. 

### Amend and Squash

You should try to keep your commit history as clear as possible, especially if you want to make a PR. This means squashing and amending commits if necessary for clarity.

## Branches

This project uses branches to keep separate spheres of development separate. There are a few predefined first level branches (though you can make custom ones too, preferably after consultation). Prefix branches with these types. If you this branch belongs to a team, add the team's name to the branch name. If it belongs to an individual contributor, add your name to it.

### Experimental

Experimental branches are for educational and prototype use. Preferably, if the mutations are not meant to persist, just fork the repo, but if the mutations are intended to persist or if you want to use this repo to share code with members of the Remanufacturing Lab and there is really no other way, Experimental branches may be used. Experimental branches should be deleted if merging will not conceivably happen. You can document the results of the experiment in `docs/experimental`.

### Feat

Feature branches are for additive mutations to existing components, replacing code with enhancements, or introducing new ones. New technologies can be added to the tech stack in these branches. These branches are intended to be turned into PRs.

### Fix

These are here to fix bugs or formatting errors that were not caught in the reviews or testing phases of the PR. These branches are intended to be turned into PRs.

### Doc

Branches where only documentation will be added and nothing else. This could be documentation of experiments, for instance. These branches are intended to be turned into PRs.

## Gitignore

Files that should always be omitted:
- Sensitive information (credentials should *never* be under version control)
- Build artefacts
- Temporary files
- Dependencies that are hosted publicly (Maven, npm, pip, etc)
- User-specific files (IDE preferences and OS files for instance)

Files that should preferably be omitted:
- Generated files (if generating them locally is easy enough)
- Prod config (if it is easy enough to write your own prod config based on dev config)

## PRs

Feat, Fix and Doc branches that have the main branch as the target branch for a merge should always resolve in a PR (pull request). Branches that merge into other branches do not have this requirement and can be merged directly, though PRs are still preferable for larger merges. Please provide a comprehensive PR message for your PR. Also provide attribution if that is not implicit from the branch name.

### Testing

PRs should have full test coverage for automated tests.

*IMPORTANT* 

Keep the testing suites up-to-date! This means that if you develop new functionality that can be tested automatically, write comprehensive tests! If you are developing a hardware-specific piece of software (like PLC code), stub it as a locally runnable program, or update the existing stub. If you do not know how to stub, please ask someone else who does, but do not compromise on test comprehensibility. 

### Code review

After all automated tests pass, PRs should be code reviewed by at least one contributor, and project management (though this may depend on the PR specifics) before being merged to main. If you are a reviewer, make sure the code works end-to-end, and read as much of the code being pushed as you can.

## Main branch

The main branch also functions as a release branch, meaning that it should be as stable as possible in production. For naming convention, use [Semantic Versioning](https://semver.org/). This branch will mostly be handled by project management.
