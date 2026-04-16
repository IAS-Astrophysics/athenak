# Introduction

<!-- This is based on the CONTRIBUDING.md document of https://github.com/PrincetonUniversity/athena/ -->

Welcome! Thank you for considering contributing to AthenaK. This project adheres to a [code of conduct](CODE_OF_CONDUCT.md), which you are expected to uphold by participating. 

The guidelines in this document are meant to help make the development of AthenaK straightforward and effective. They are a set of best practices, not strict rules, and this document may be modified at any time. Navigating the code can be daunting for new users, so if anything is unclear, please let us know!

<!-- ### Table of Contents -->


# How to contribute
There are many ways to contribute! We welcome feedback, [documentation](#documentation), tutorials, scripts, [bug reports](#bug-reports), [feature requests](#suggesting-enhancements), and [quality pull requests](#pull-requests).

## Using the issue tracker
Both [bug reports](#bug-reports) and [feature requests](#suggesting-enhancements) should use the [GitHub issue tracker](https://github.com/PrincetonUniversity/athena/issues).

Please do not file an issue to ask a question on code usage.

### Bug reports
[Open a new Issue](https://github.com/IAS-Astrophysics/athenak/issues/new)

### Suggesting enhancements
Feature requests are welcome, and are also tracked as [GitHub issues]( https://guides.github.com/features/issues/).

Please understand that we may not be able to respond to all of them because of limited resources.

## Submitting changes
Some requirements for code submissions:
- AthenaK is licensed under the BSD 3-Clause License; contributions must also use the BSD-3 license.
- The code must be commented and well documented, see [Documentation](#documentation).
- AthenaK code follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). Please follow these conventions as closely as possible in order to promote consistency in the codebase.
- When implementing new functionality, add a regression test. See [Testing and continuous integration (CI)](#testing-and-continuous-integration-CI).
- If your submission fixes an issue in the [issue tracker](https://github.com/IAS-Astrophysics/athenak/issues), please reference the issue # in the pull request title or commit message, for example:
```
Fixes #42
```

The below instructions assume a basic understanding of the Git command line interface.
If you are new to Git or a need a refresher, the [Atlassian Bitbucket Git tutorial](https://www.atlassian.com/git/tutorials/learn-git-with-bitbucket-cloud) and the [Git documentation](https://git-scm.com/) are helpful resources.

The easiest way to contribute to Athena++ is to fork the repository to your GitHub account, create a branch on your fork, and make your changes there. When the changes are ready for submission, open a pull request (PR) on the Athena++ repository. The workflow could be summarized by the following commands:
1. Fork the repository to your GitHub account (only once) at https://github.com/PrincetonUniversity/athena/fork
2. Clone a local copy of your fork:
```
git clone https://github.com/<username>/athenak ./athenak-<username>
```
3. Create a descriptively-named feature branch on the fork:
```
cd athenak-<username>
git checkout -b cool-new-feature
```
4. Commit often, and in logical groups of changes.
  * Use [interactive rebasing](https://help.github.com/articles/about-git-rebase/) to clean up your local commits before sharing them to GitHub.
  * Follow [commit message guidelines](https://www.git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project#_commit_guidelines); see also [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/).
```
git add src/modified_file.cpp
# Use your editor to format the commit message
git commit -v
```
5. Push your changes to your remote GitHub fork:
```
git push -u origin cool-new-feature
```
6. When your branch is complete and you want to add it to AthenaK, [open a new pull request to `main`](https://github.com/PrincetonUniversity/athena/pull/new/main).

### Forks and branches
The use of separate branches for both new features and bug fixes, no matter how small, is highly encouraged. Committing directly to `main` branch should be kept to a minimum. [Branches in Git are lightweight](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell), and merging small branches should be painless.

For the majority of development, users should use personal forks instead of branches on [IAS-Astrophysics/athenak](https://github.com/IAS-Astrophysics/athenak) (especially for larger development projects). The shared AthenaK repository should only contain a restricted set of main feature branches and temporary hotfix branches at any given time. <!-- consider reaching out to AthenaK developers before starting any significant PR/feature development to see if anyone is working on it or if we would consider merging it into AthenaK-->

To update your fork with changes from [IAS-Astrophysics/athenak](https://github.com/IAS-Astrophysics/athenak), from the `main` branch on a cloned copy of the forked repo:
1. Add a remote named `upstream` for the original Athena++ repository:
```
git remote add upstream https://github.com/IAS-Astrophysics/athenak
```
2. Fetch the updates from the original Athena++ repository:
```
git fetch upstream
```
3. Merge the new commits into your forked `main`:
```
git merge --ff-only upstream/main
```
will work if you have not committed directly to your forked `main` branch.

If you have modified your forked `main` branch, the last two steps could be replaced by:
```
git pull --rebase upstream main
```
See [Developing on shared `branch`](#developing-on-shared-branch).

### Developing on shared `branch`
There are a few practices that should be followed when committing changes to a collaborative `branch` on [IAS-Astrophysics/athenak](https://github.com/IAS-Astrophysics/athenak) in order to avoid conflicts and headaches. These guidelines especially apply to developing on the fast changing `main` branch for those users with Admin permissions.

If you commit to an outdated local copy of `branch` (i.e. someone else has pushed changes to GitHub since you last checked), the `git push origin branch` command will be rejected by the server and prompt you to execute the `git pull` command. The default `git pull` behavior in this scenario is to create a merge-commit after you resolve any conflicts between your changes and the remote commits. However, these non-descriptive commit messages tend to clutter the repository history unnecessarily.
<!-- insert image of Network graph to compare linear and non-linear Git history -->

For example, searching the Athena++ repository history using the [GitHub website](https://github.com/PrincetonUniversity/athena/search?utf8=%E2%9C%93&q=merge+branch+%27master%27+of+https:&type=Commits) or the command line:
```
git log --oneline —grep="Merge branch ‘master' of https://github.com/PrincetonUniversity/athena$" | wc -l
```
returns many such commits. Most of them likely could have been avoided by either 1) doing local development on feature branches or 2) using `git pull --rebase` to perform a rebase instead of a merge when pulling conflicting updates.

If you frequently encounter such issues, it is recommended to enable the latter by default. In git versions >= 1.7.9, this can be accomplished with:
```
git config --global pull.rebase true
```

### Pull requests
When your changes are ready for submission, you may open a new pull request to `main` [from a branch on the main repository (Write access)](https://github.com/IAS-Astrophysics/athenak) or from a branch on your forked repository. For the latter, go to the page for your fork on GitHub, select your development branch, and click the pull request button.

We will discuss the proposed changes and may request that you make modifications to your code before merging. To do so, simply commit to the feature branch and push your changes to GitHub, and your pull request will reflect these updates.

Before merging the PRs, you may be asked to squash and/or rebase some or all of your commits in order to preserve a clean, linear Git history. We will walk you through the interactive rebase procedure, i.e.
```
git rebase -i main
```

In general for AthenaK, merging branches with `git merge —no-ff` is preferred in order to preserve the historical existence of the feature branch.

After the pull request is closed, you may optionally want to delete the feature branch on your local and remote fork via the GitHub PR webpage or the command line:
```
git branch d cool-new-feature
git push origin --delete cool-new-feature
```

### Code review policy
Currently, `main` is a GitHub [protected branch](https://help.github.com/articles/about-protected-branches/), which automatically:
* Disables force pushing on `main`
* Prevents `main` from being deleted

Additionally, we have enabled ["Require pull request reviews before merging"](https://help.github.com/articles/enabling-required-reviews-for-pull-requests/) to `main`. This setting ensures that all pull requests require at least 1 code review before the branch is merged to the `main` branch and effectively prohibits pushing **any** commit directly to `main`, even from users with Write access. Attempting to do so will result in an error such as:
```
Total 9 (delta 7), reused 0 (delta 0)
remote: Resolving deltas: 100% (7/7), completed with 7 local objects.
remote: error: GH006: Protected branch update failed for refs/heads/main.
remote: error: At least 1 approving review is required by reviewers with write access.
To git@github.com:https://github.com/IAS-Astrophysics/athenak.git
! [remote rejected] main -> main (protected branch hook declined)
error: failed to push some refs to 'https://github.com/IAS-Astrophysics/athenak.git'
```

Only collaborators with Admin permissions can bypass these restrictions. The decision to force the use of branches and pull requests for all changes, no matter how small, was made in order to:
1. Allow for isolated testing and human oversight/feedback/discussion of changes
2. Promote a [readable](https://fangpenlin.com/posts/2013/09/30/keep-a-readable-git-history/), [linear](http://www.bitsnbites.eu/a-tidy-linear-git-history/), and reversible Git history for computational reproducibility and maintainability
3. Most importantly, prevent any accidental pushes to `main`
<!-- Currently set # of required reviews to 1; other options to consider enabling in the future include: -->
<!-- "Dismiss stale PR approvals when new commits are pushed" -->
<!-- "Restrict who can push to this branch" (redundant with Require PR reviews)-->
<!-- "Require status checks to pass before merging" after separating CI build steps in new GitHub Checks API-->
<!-- "Include administrators" -->

When anyone opens a new pull request to `main`, GitHub will automatically request a code review from one or more users defined by the PR's modified files and the rules in the current [`.github/CODEOWNERS`](https://github.com/PrincetonUniversity/athena/blob/main/.github/CODEOWNERS) file. Only users with Admin permissions may modify this file to designate collaborators with at least Write access as "code owners". It is possible to use separate versions of this file on each branch to regulate PRs targeting those branches; see "[About CODEOWNERS](https://help.github.com/articles/about-codeowners/)" for more information.

## Testing and continuous integration (CI)
Automated testing is an essential part of any large software project. Regression tests are included in `tst/` folder. These are run automatically on every pull request. 

## Documentation
The development repository's [documentation](https://github.com/IAS-Astrophysics/athenak/wiki) is a [GitHub Wiki](https://help.github.com/articles/about-github-wikis/) and is written largely in Markdown. Limited math typesetting is supported via HTML. See existing Wiki source for examples.

Any significant change or new feature requires accompanying documentation before being merged to `main`. While edits can be made directly using the online interface, the Wiki is a normal Git repository which can be cloned and modified. [However](https://help.github.com/articles/adding-and-editing-wiki-pages-locally/):
> You and your collaborators can create branches when working on wikis, but only changes pushed to the `main` branch will be made live and available to your readers.

## Community
The AthenaK private Zulip workspace is located at [athena-k.zulipchat](https://athena-k.zulipchat.com/). Issues and pull requests on the GitHub repository should still be the main forum to discuss development details, but the Zulip workspace is a useful centralized forum for general discussion, sharing new results, asking questions, and learning what others are working on.

At this time, the Zulip workspace is closed to the general public.

## Versioning and releases
We intend to provide periodic releases, versioned according to CalVer, or [Calendar Versioninng](https://calver.org/). A detailed walkthrough of the steps a project maintainer must complete in order to mint a new release is provided in the following section.

We currently maintain Git tags and code versions in a one-to-one correspondence: all versions are tagged, and all tags have a version number. However, not all tags/versions are released (see below section discussing pre-release tagged versions). A release version is defined by drafting a [GitHub Release](https://help.github.com/articles/creating-releases/) along with release notes in the GitHub UI. 

Each release is accompanied by an Git annotated (not lightweight) tag. An annotated tag is a full Git object with its own tagger name, tagger email, and creation date. A lightweight Git tag is more appropriate for temporary or local/personal use than for publishing releases, since a lightweight tag is merely a pointer to a commit object (much like a branch that doesn't naturally move with commits and *shouldn't* be moved by users after it is shared). Therefore, the tag should be created from the Git CLI, not the GitHub UI which only supports creating lightweight tags as of 5/24/18.