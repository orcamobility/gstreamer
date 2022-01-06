---
short-description: Contributing to GStreamer
...

# Contributing to GStreamer

This document provides instructions and guidelines for submitting issues,
feature requests and patches to GStreamer. The following applies to all
these operations:

- Please use [freedesktop.org GitLab][gitlab] to perform any of the aforementioned
  operations. You will need to create a freedesktop.org GitLab account if you
  don't have one yet (yep, that's just how it is. Sorry for the inconvenience)
  If you don't want to create a new account you should also be able to sign in
  with a Google, GitHub or Twitter account.

## How to File Issues and Request for Enhancements

### Where to File Issues and Feature Requests

- Create a new issue if there is no existing report for this problem yet.
  The GStreamer [bugs page][bugs] also has shortcuts for the major components
  and simple search functionality if you'd like to browse or search for
  existing issues, or use the [GitLab search bar for the GStreamer project][gitlab]
  at the top of the page.

- If you are filing a feature request (i.e. anything that is not supposed to
  work already, that is anything not an issue), please add the *Enhancement* label.
  Feel free to add any other appropriate already existing labels. Please don't
  create new labels just for your issue. This won't affect the way we prioritise
  the issue, but it will make triaging easier for us. In particular, do not add
  the *Blocker* label to an issue just because an issue is important to you.
  This label should be added only by GStreamer maintainers.

- If your issue is about a specific plugin, element or utility library,
  please prefix the issue summary with `element-name:`, `plugin-name:` or `lib:`
  and keep the rest of the description as short and precise as possible.

  Examples:

   - `id3demux: fails to extract composer tags`
   - `tsdemux: does not detect audio stream`
   - `Internal flow error when playing matroska file`

  This makes sure developers looking through the list of open issues or issue
  notification mails can quickly identify what your issue is about. If your text
  is too long and only contains fill words at the beginning, the important
  information will be cut off and not show up in the list view or mail client.

- If you don't know which component to file the issue against, just pick the one
  that seems the most likely to you, or file it against the gstreamer-project
  component. If in doubt just pop into our IRC channel `#gstreamer` on the
  [OFTC IRC network](https://www.oftc.net/), which you can connect to using
  any IRC client application or the [OFTC IRC webchat](https://webchat.oftc.net/?channels=%23gstreamer).
  In any case, if it's not the right component someone will move the issue
  once they have a better idea what the problem is and where it belongs.

- Please mention:

   - what version of GStreamer you are using
   - what operating system you are using (Windows, macOS, Linux)
   - if you're on Linux, please mention your distro and distro version
   - if this is on an embedded device please provide details

- Try to describe how the issue can be reproduced. If it is triggered by any
  specific file, try to make the file available somewhere for download and
  put the link into the issue. The easier it is for us to reproduce
  the issue, the easier it is to fix it.

- If you experience a crash (that is: the application shuts down unexpectedly,
  usually with a segfault or bus error or memory access violation or such),
  please try to [obtain a stack trace][stack-trace]. If there are criticals
  or warnings printed right before the crash, run with the environment variable
  `G_DEBUG=fatal_warnings` set, then it will abort on the first warning, which
  should hopefully give an indication to where the problem is. You can then
  obtain a stack trace from where it aborts.

- If the application errors out, please provide a gst debug log. You can get
  one by setting the `GST_DEBUG=*:6` environment variable, combined with
  `GST_DEBUG_FILE=/tmp/dbg.log`. The resulting file might end up being very
  large, so it's advisable to compress it with `xz -9 /tmp/dbg.log` before
  sharing. You may also be asked to provide debug logs for specific debug
  categories rather than everything (`*:6`).

[stack-trace]: https://wiki.gnome.org/Community/GettingInTouch/Bugzilla/GettingTraces/Details

## How to Submit Patches

### Where to Submit Patches

Patches need to be submitted through [GitLab][gitlab] in form of a
"Merge Request" (MR), which is the same as a "Pull Request" (PR)
in GitHub, and uses the same [workflow][gitlab-merge-request-workflow].

[gitlab-merge-request-workflow]: https://docs.gitlab.com/ce/user/project/merge_requests/index.html#overview

In order to submit a merge request you first need a personal fork of the
project / gstreamer module in question. To fork a module go to
the module in question (e.g. <https://gitlab.freedesktop.org/gstreamer/gstreamer>)
and hit the fork button. A new repository will be created in your user namespace
(<https://gitlab.freedesktop.org/$USERNAME/gstreamer>). You will be redirected
there automatically once the forking process is finished. For big repositories
the forking might take a few minutes.

Once this is done you can add your personal fork as new remote to your existing checkout with

    git remote add my-personal-gitlab-fork git@gitlab.freedesktop.org:$USERNAME/gstreamer.git

Check with

    git fetch my-personal-gitlab-fork

that it is accessible and working.

If you have not done so already, you may need to first
[set up SSH keys in your GitLab User Settings](https://gitlab.freedesktop.org/profile/keys).

**Once you have done that, please make your repository public.** By default
newly-forked repositories might be private, but that causes problems for
maintainers and merge bots, so please go to the newly-created repository's
settings (https://gitlab.freedesktop.org/$USERNAME/gstreamer/edit) and set
the visibility to public. Thanks!

Next, you make a git branch with one or more commits you want to submit
for review and merging. For that you will first need a local branch which
you can create with e.g.

    git checkout -b fix-xyz

You can then push that branch to your personal fork git repository with

    git push my-personal-gitlab-fork

You can use

    git push --dry-run my-personal-gitlab-fork

for testing to see what would happen without actually doing anything yet.

After you have pushed the branch to your personal fork you will see a link
on the terminal with which you can create a merge request for the upstream
repository. You can also do this later by going to the
branch list of your personal repository at
<https://gitlab.freedesktop.org/$USERNAME/gstreamer/branches>
and then hitting the 'Merge Request' button when ready. This will open a new
page where you can enter a description of the changes you are submitting.

Couple of additional points:

- If you are submitting a Merge Request for an issue (or multiple issues) that
  already exist, please add 'Fixes #123' to the commit message of one of your
  commits, so that there is a cross-reference in GitLab and the issue will be
  closed automatically when your Merge Request is merged.

- You do not have to file an issue to go with each Merge Request, it's fine
  to just submit a Merge Request on its own.

- **Please make sure the "Allow commits from members who can merge to the target branch"**
  **checkbox is enabled when submitting merge requests** as otherwise
  maintainers (and our merge bot) can't rebase your Merge Request when they
  want to merge it, which means they won't be able to merge it. If you can't
  enable the check box that means your personal repository fork is private.
  In that case, please go to the settings and change visibility to public.

- If your proposed changes are proposed for review but not ready to be merged
  yet, please prefix the Merge Request title with `WIP:` for Work-in-Progress.
  That will prevent us from inadvertently merging it and make clear its status.

- Please make sure the 'Author' field in your commit messages has your full and
  proper name and e-mail address. You can check with e.g. `git log` or `gitk`.

- If your change is for an enhancement (anything that is not supposed to work
  already, i.e. anything not a bug) or adds new API, please add the
  *Enhancement* label. This won't affect the way we prioritise your issue,
  but it does make triaging easier for us.

- If your Merge Request is against a specific plugin or element or utility library,
  please prefix the Merge Request summary with `element-name:`, `plugin-name:`
  or `lib:` and keep the rest of the description as short and precise as possible.

  Examples:

   - `id3demux: add support for WCOP frame`
   - `riff: add RGB16 support`
   - `playbin: detect if video-sink supports deinterlacing`
   - `tests: rtprtx unit test is racy`

  This makes sure developers looking through the list of open merge requests or
  notification mails can quickly identify what your change is about. If your text
  is too long and only contains fill words at the beginning, the important
  information will be cut off and not show up in the list view or mail client.

- Make liberal use of the reference syntax available to help cross-linking
  different issues and merge requests. e.g. `#100` references issue 100 in the
  current module. `!100` references merge request 100 in the current project.
  A complete list is available from [gitlab's documentation][special-md-references].

- Please create separate merge requests for separate issues and for different
  modules. There is no golden rule when something counts as a separate issue,
  please just use your best judgment.  If a merge request is related to another
  merge request in another module please mention that in the description using
  a gitlab reference as outlined above.  For example, if you have a change that
  needs to be done in each module, one issue with one merge request per module
  is fine. If there is an issue that requires related fixes in multiple elements
  or libraries, please also feel free to put everything into one issue. If you
  just happen to have multiple patches for us but they are not really related,
  please put them in separate issues and merge requests. The main question is
  if it makes sense to discuss and review these patches together or if they
  could just as well be handled completely separately.

- Please do not send patches to the gstreamer-devel mailing list. Patches
  submitted on the mailing list are most likely going to be ignored, overlooked,
  or you will get a brief reply asking you to put them into gitlab. We do
  not use the mailing list for patch review.

- Please do not send pull requests to our github mirror. They will be closed
  automatically.

- Please do not attach patches to existing bugs on [GNOME Bugzilla][bugzilla]
  If you want to reopen an already closed bug, let one of the developers know
  and we will look into that on a case-by-case basis.

- Please do not attach patches to issues.


[special-md-references]: https://docs.gitlab.com/ee/user/markdown.html#special-gitlab-references
[bugzilla]: https://bugzilla.gnome.org/
[bugs]: https://gstreamer.freedesktop.org/bugs/
[gitlab]: https://gitlab.freedesktop.org/gstreamer

### How to Prepare a Merge Request for Submission

Please prepare any merge request against a current git checkout of the
GStreamer monorepo (gstreamer module), against the tip of the `main` branch.
If a merge request is prepared against an old commit or older branch and can't
be easily rebased you may be asked to rebase and update the branch on top of
the `main` branch.

If you have created a new plugin, please submit a merge request that adds it to
`subprojects/gst-plugins-bad`, including any required `meson.build` modifications,
new files and documentation.

#### Preparation: create a personal fork of the git repository on GitLab

The easiest way to create a merge request is to create one or more local commits
for your changes in a branch in a local git repository. This should be a git
clone checkout of your fork of the module in question.  To fork a module go to
the module in question (e.g. <https://gitlab.freedesktop.org/gstreamer/gstreamer>)
and hit the fork button.  A new repository will be created in your user namespace
and should be accessible as
<https://gitlab.freedesktop.org/$USERNAME/gstreamer>.

Clone the main gstreamer repository, like this:

    git clone https://gitlab.freedesktop.org/gstreamer/gstreamer.git

and then add your personal fork as a git remote like this:

    git remote add my-personal-gitlab-fork git@gitlab.freedesktop.org:$USERNAME/gstreamer.git

Check with

    git fetch my-personal-gitlab-fork

that it is accessible and working.

If you have not done so already, you may need to first
[set up SSH keys in your GitLab User Settings](https://gitlab.freedesktop.org/profile/keys).

#### Creating a branch for the merge request and adding commits to it

Once you have a git repository with the original code in it, you should create a
branch for your change. e.g. to create a branch and checkout:

    git checkout -b topic-branch

Then you can make your modifications and create a local commit with e.g.

    git commit path/to/file1.[ch]

This will pop up an editor where you can create your commit message. It should
look something like:

    exampledemux: fix seeking without index in push mode

    Without an index we would refuse to seek in push mode. Make
    seeking without an index work by estimating the position
    to seek to. It might not be 100% accurate, but better than
    nothing.

Then exit the editor, and you should have a commit.

It's best to run `git add` or `git commit` on specific directories or files
instead of using `git commit -a`, as it's too easy to accidentally contaminate
a patch with changes that belong into it with `git commit -a`, in particular
changes to the `common` submodule.

You can check your commit(s) with `git show` or `git log -p` or using a GUI
such as `gitg` or `gitk`.

Make sure the author is correctly set to your full name and e-mail address.

If you haven't used git before, it would be a good idea to tell it who you are:

    $ git config --global user.name "George S. Treamer"
    $ git config --global user.email "george.s.treamer@example.com"

You can make changes to the last commit using:

 - `git commit --amend` to fix up the commit message

 - `git commit --amend --author='John You <john@you.com>'` to fix up the author

 - `git add path/to/file1.[ch]; git commit --amend` to incorporate fixes
    made to the files since the last commit (i.e. what shows up in `git diff`).
    If you just want to add some of the changes, but not all of them you can
    use `git add -p file.c`, then it will ask you for each individual change
    whether you want to add it or leave it.

Once everything looks fine, push your branch to your local fork e.g. using

    git push your-personal-gitlab-fork your-branch

This push will display a link to be able create a merge request from your branch.
Click this link and fill out the details of the merge request.  You can also
create a merge request from an existing branch. See the
[gitlab documentation][create-mr] for more details.

Please make sure your commits are as terse and precise as possible. Do not
include 'clean-ups' or non-functional changes, since they distract from the
real changes and make things harder to review, and also lower the chances that
the patch will still apply cleanly to the latest version in git. If you feel
there are things to clean up, please submit the clean-ups as a separate patch
that does not contain any functional changes.

[create-mr]: https://docs.gitlab.com/ee/gitlab-basics/add-merge-request.html

#### Coding Style

Try to stick to the GStreamer indentation and coding style. There is a script
called [`gst-indent`][gst-indent] which you can run over your `.c` or `.cpp`
files if you want your code auto-indented before making the patch. The script
requires GNU indent to be installed already. Please do _not_ run `gst-indent` on
header files, our header file indentation is free-form. If you build GStreamer
from git, a local commit hook will be installed that checks if your commit
conforms to the required style (also using GNU indent).

Different versions of GNU indent may occasionally yield slightly different
indentations. If that happens, please ignore any indentation changes in
sections of code that your patch does not touch. You can do that by staging
changes selectively via `git add -p`. You can bypass the local indentation
check hook by using `git commit -n`, but it will still be checked again later
when you submit your changes through GitLab for merging.

We are working on making this less hasslesome.

[gst-indent]: https://gitlab.freedesktop.org/gstreamer/gstreamer/-/blob/main/scripts/gst-indent-all

Compiler requirements:
 - we are targetting the C99 compiler and preprocesser feature subset
   supported by gcc 5.4, clang and vs2017. Newer features can be
   used conditionally as long as there is a suitable fallback. This is similar
   to [GLib's compiler requirements](https://wiki.gnome.org/Projects/GLib/CompilerRequirements).

Other style guidelines:
 - do not use tabs for indentation in new code, not even in freeform header files
 - do not use packed structures for parsing external data, instead use helper
   APIs such as [GstByteReader][bytereader] and [GstBitReader][bitreader] for
   data parsing
 - from GStreamer 1.20 onwards you may in particular also
   - use c++-style `// comments`
   - declare variables inline (as opposed to only at the beginning of a block)
   - use advanced/nicer struct initialisers

[gst-indent]: https://gitlab.freedesktop.org/gstreamer/gstreamer/tree/master/tools/gst-indent
[bitreader]: https://gstreamer.freedesktop.org/documentation/base/gstbitreader.html?gi-language=c#GstBitReader
[bytereader]: https://gstreamer.freedesktop.org/documentation/base/gstbytereader.html?gi-language=c#GstByteReader

### Writing Good Commit Messages

Please take the time to write good and concise commit messages.

The first line of each commit message should be a short and concise summary
of the commit. If the commit applies to a specific subsystem, library, plugin
or element, prefix the message with the name of the component, for example:

    oggdemux: fix granulepos query for the old theora bitstream

or

    docs: add new stream API

or

    tests: video: add unit test for converting RGB to XYZ colorspace

This should be a *summary* of the change and _not a description_ of the change.
Meaning: don't say *how* you did something but *what* you fixed, improved or
changed, what the most important practical *effect* of the change is. Example:

    qtdemux: fix crash when doing reverse playback in push mode (good)

instead of

    qtdemux: use signed integer to avoid counter underrun (bad)

The second line of the commit message should be empty.

The third and following lines should contain an extensive *description* and
*rationale* of the change made: what was changed, what was broken, how did it
get fixed, what bugs or issues does this fix? And most importantly: *why* was
something changed.

Trivial commits do not require a description, e.g. if you fix a memory leak
it's usually enough to just say that you fixed a leak. Maybe mention what was
leaked and perhaps also if it was an important leak or only happens in some
corner case error code path, but in any case there's no need to write a long
explanation why leaks are bad or why this needed fixing.

The important part is really what the reasoning behind the change is, since
that's what people want to know if they try to figure out twelve months later
why a line of code does what it does.

If the commit is related to any particular issues in gitlab, please add a
reference to the issue (e.g. `See #123` or `Fixes #123` if it fixes it the
issue). For issues in other repositories (gst-plugins-{base,good,ugly,bad} etc.)
please add the full issue URL to the commit message instead (or ask for the
issue to be moved to the monorepo gstreamer repository), e.g.

    https://gitlab.freedesktop.org/gstreamer/gst-plugins-foo/issues/123

We do not use `Signed-off by:` lines in GStreamer, please create commits
without such lines.

Please do not add references to private company-internal bug trackers or
code repositories in commit messages.

### After Submitting your Merge Request

Whenever you submit a new Merge Request, add a comment to an existing issue or
Merge Request, GitLab will send a notification e-mail to GStreamer
developers. This means that there is usually no need to advertise the fact that
you have done so in other forums such as on IRC or on the mailing list, unless
you have been asked to file an issue there, in which case it's nice to follow up
with the link to the issue.

Most of all, please be patient.

We try to review patches as quickly as possible, but there is such a high
volume of issues, merge requests and feature requests that it is not always
possible to tend to them all as quickly as we'd like. This is especially
true for completely new plugins or new features.

If you haven't received any response at all for a while (say two weeks or so),
do feel free to ping developers by posting a quick follow-up comment on the
issue or merge request.

If you do not get a response, this is usually not a sign of people *ignoring*
the issue, but usually just means that it's fallen through the cracks or
people have been busy with other things.

Most GStreamer developers have a review workflow that's driven by e-mail or
GitLab notifications, so posting a follow-up comment is the best way to draw
attention to an issue or merge request.

### Updating Your Merge Request and Addressing Review Comments

When someone reviews your changes, they may leave review comments for
particular sections of code or in general. These will usually each start
a new "Discussion" which is basically a thread for each comment.

When you believe that you have addressed the issue raised in a discussion,
either by updating the code or answering the questions raised, you should
"Resolve the Discussion" using the button, ideally also leaving a comment
saying so (e.g. "done", "fixed", "updated", "no longer needed" or such). The
comment makes sure a notification e-mail is generated, which makes it easier
for GStreamer developers to keep track of what's happening.

At the top of each merge request in GitLab is a tracker with the number of
unresolved discussions. This way it's easy for maintainers (and yourself)
to see what's left to do and if there are any open questions/issues.

Whenever you have made changes to your patches locally you can just
`git push -f your-personal-gitlab-fork your-branch` to your personal fork.
GitLab will then pick up the changes automatically. You do _not_ need to submit
a new Merge Request whenever you make changes to an already-submitted patchset,
and in fact you shouldn't do that because it means all the previous discussion
context is lost and it's also not easy for reviewers to see what changed.
Just force-update your existing branch.

You do not need to add individial "fixup commits" to your branch when you address
issues raised. Instead just fix up the original commit(s) directly using
`git rebase` etc. GitLab is able to track and show changes made between
different revisions of a merge request branch, so just keep the branch always
to the "latest clean version". See the [Rewriting History](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)
section of the [Pro Git book](https://git-scm.com/book/en/v2) for more details.

GStreamer maintainers will typically receive e-mail notifications when you add
a comment and when all oustanding discussions have been resolved. They may or
may not receive e-mail notifications when you update the commits in your branch.

# Workflows for GStreamer developers

## Backporting to a stable branch

Before backporting any changes to a stable branch, they should first be
applied to the `main` branch, and should obviously not have caused any known
outstanding regressions. The only exception here are changes that do not apply
to the `main` branch anymore.

You do not need to create backport merge requests against stable branches.
Backport merge requests for stable branches will be created automatically
based on labels on `main` branch merge requests.

Existing merge request against the `main` branch, including merged ones,
that should be considered for backporting in the future should be labeled with
the `Needs backport` label and the label for the target stable branch
(e.g. `1.18`). All merge requests with the [`Needs backport`][needs-backport]
label will be regularly considered for backporting by GStreamer developers.

Changes that add API or change existing API will usually not be considered
for backporting.

In case patches don't apply cleanly to the target stable branch, draft `WIP`
merge requests will be created by the above-mentioned scripts. If that happens
developers will have to backport the desired commits and resolve any cherry-pick
conflicts manually. The result should then be pushed into the existing branch
for the `WIP` backport merge request. Any GStreamer developer should be able
to do this, and help in this area is always welcome. Please resolve the WIP/Draft
status once all the relevant commits have been backported and assign the
merge request to the merge bot for merging.

There is also a [`Maybe backport` label][maybe-backport] which can be added
if you're not sure if commits should be backported or not, or you want to
revisit it later once the changes in question have seen some testing in the
development branch. If you tag a merge request with `Maybe backport` it's
usually useful to add a comment about why it's being tagged only as a maybe.
That will make it easier for others to evaluate whether it should be backported
or not later.

[needs-backport]: https://gitlab.freedesktop.org/groups/gstreamer/-/merge_requests?state=merged&label_name[]=Needs%20backport
[maybe-backport]: https://gitlab.freedesktop.org/groups/gstreamer/-/merge_requests?state=merged&label_name[]=Maybe%20backport
