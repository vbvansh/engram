# Problems and Fixes

This page lists the problems found when Engram's memory was tested on the
LoCoMo question set (1,540 answerable questions about 10 long conversations),
and how to fix each one. The numbers come from that test.

## Summary

| # | Problem | How much it costs | Fix in short |
|---|---|---|---|
| 1 | Found memories are thrown away | 149 questions, all answered wrong | Keep found memories when their label does not match — **fixed** (see result below) |
| 2 | False "conflicting evidence" alarms | 95 questions, all answered wrong | Only call it a conflict for facts that can have one value — **fixed** (see result below) |
| 3 | Dates stay as "last week" | Date questions: only 24% correct | Turn relative times into real dates when storing |
| 4 | Facts get wrong or vague labels | Feeds problem 1; template answers only 10% correct | A label list for everyday life, fixed synonyms |
| 5 | Other users' memories crowd out search results | Search missed 8.5% of the right memories; no score change | Filter by user inside the search, not after it — **fixed** (see result below) |
| 6 | Messages are stored without their context | Not yet measured | Send earlier turns; don't drop a whole message |

---

## 1. Found memories are thrown away

**What happens.** Some words in a question ("work", "job", "favourite",
"team", "live", "move", "attend", "born", …) make Engram decide in advance
what *label* the answer must have — for example "works at". It then keeps only
facts with exactly that label and throws everything else away, even the right
memory it just found. The reply becomes the fixed sentence *"I don't have
enough verified memory evidence to answer that."*

The expected label can also be invented by the AI planner: for "What is the
name of Lena's cat?" it asked for a "has pet" fact, while the memory said
"Lena owns Biscuit".

**Example.** "When did Melanie go to the pottery **work**shop?" — the word
"work" inside "workshop" makes Engram expect a "works at" fact. The memory
about the workshop is found, then thrown away.

**Cost.** 149 of 1,540 questions (9.7%). All answered wrong.

**Fix.**
- Match keywords as whole words only ("workshop" is not "work").
- Use the expected label to *rank* results, not to *remove* them.
- If no fact has the expected label, give the answer model the memories that
  were found instead of refusing.

**Result — fixed.** Keywords now match whole words, and when no fact has the
expected label after every search route, Engram answers from the memories it
found. Measured on all questions, compared one by one with the run before:

| | Before | After |
|---|---|---|
| Refusals caused by this problem | 194 | 9 |
| The 149 affected questions answered correctly | 0 | 78 |
| Strict score (1,540 questions) | 36.4% | 41.1% (+4.7, not luck) |
| J score (1,540 questions) | 48.5% | 53.3% (+4.7, not luck) |
| Trick (adversarial) questions handled correctly | 68.6% | 66.6% (−2.0, within luck) |
| Correct answers over all 1,986 questions | 867 | 930 (+63) |

All other questions changed only as much as they do by chance, so the gain
comes from the questions this fix targeted. The small drop on trick questions
was expected: some of them were "correct" only because Engram refused
everything.

## 2. False "conflicting evidence" alarms

**What happens.** Before answering, Engram checks whether its facts contradict
each other. It calls it a conflict whenever one person has two different
values for the same kind of fact. But many facts can have several true values
at the same time — a person can be in two teams, have three children, or enjoy
several hobbies. Engram then refuses with *"I found conflicting verified
memory evidence, so I cannot select one current answer."*

Because stored facts have no dates (problem 3), every fact also looks true
"forever", so old and new values always seem to clash.

**Example.** "How does John say his team handles tough opponents?" — John is
a member of more than one team, so Engram sees a "conflict" and refuses.

**Cost.** 95 of 1,540 questions (6.2%). All answered wrong.

**Fix.**
- Only flag a conflict for facts that can have one value at a time (current
  job, home city). Engram's label list already records which facts allow one
  value and which allow many — use it.
- For facts that allow many values, show all of them to the answer model.
- Once facts have dates (problem 3), compare the dates: an old job followed by
  a new job is history, not a conflict.

**Result — fixed.** Engram now calls two facts a conflict only when the
question asks about one kind of fact, that kind can hold just one value at a
time (a current job, a home city), and the found facts of that kind disagree.
A clash in some other fact (Calvin's city) no longer blocks a question about
something else (his goals). Conflicts the database itself marked still stop
the answer. Measured on all questions, compared one by one with the run after
fix 1:

| | Before | After |
|---|---|---|
| Refusals caused by this problem | 111 | 0 |
| The 88 affected questions answered correctly | 1 | 29 |
| Strict score (1,540 questions) | 41.1% | 45.5% (+4.4, not luck) |
| J score (1,540 questions) | 53.2% | 56.8% (+3.6, not luck) |
| Trick (adversarial) questions handled correctly | 66.6% | 66.6% (no change) |
| Correct answers over all 1,986 questions | 930 | 998 (+68) |

The gain has two sources. The affected questions gained 28 and lost none.
Questions that used to search further after a false conflict now answer at
the first step, and gained 28 while losing 10. The remaining questions changed
about as much as they do by chance. The 24 affected trick questions lost 9, as
expected: they were "correct" only because Engram refused all of them.

## 3. Dates stay as "last week"

**What happens.** When a conversation is stored, the date of each message is
not passed to the step that pulls facts out of it. So "I went hiking last
week" is stored with the words "last week" instead of a real date. Each fact
is also stamped with the day it was *stored*, not the day it was *said*.

**Example.** "When did Maria join a gym?" — Engram answers "Maria joined a gym
last week." The correct answer is "the week before 16 June 2023".

**Cost.** Date questions are the weakest group: only 24% correct. It also
causes many of the false conflicts in problem 2.

**Fix.**
- Give the fact-extraction step the date of each message.
- Ask it to turn relative times ("yesterday", "last week", "next month") into
  real dates.
- Store that date with each fact ("true from"), and the message time as
  "said at".
- The conversations must be stored again after this change.

## 4. Facts get wrong or vague labels

**What happens.** Engram's built-in label list knows only 12 kinds of facts,
mostly about work (works at, manager, role, team). Any other fact keeps
whatever label the AI made up ("inspired by", "busy with",
"has favourite colour"), so similar facts with different labels never match.
Some synonyms are wrong: "attended" always means "attends school", so
"Caroline attended a support group" is stored as *"Caroline attends school:
LGBTQ support group"*.

**Cost.** It feeds problem 1 (labels that never match), and the short template
answers Engram builds from these labels are right only 10% of the time. After
fix 2 more questions reach these templates: 59 template answers, and only 3 of
the 43 on normal questions are right — for example "nate room attended
game_convention and convention." for "When did Nate attend a cooking show?".

**Fix.**
- Add labels for everyday life: likes, hobbies, family, pets, events attended,
  trips, health, plans.
- Fix wrong synonyms ("attended an event" is not "attends school").
- Group similar labels so they count as the same (has favourite colour =
  prefers).

## 5. Other users' memories crowd out search results

**What happens.** All users share one search index. A search first picks the
18 memories closest to the question from *everyone*, and only afterwards
removes the ones that belong to other users. If other users have similar
memories, this user may get only a few results — or none.

**Cost.** Affects every question, and a user's results change depending on
how much *other* data is stored. Measured with the 10 conversations stored:
the search found only 91.5% of the 12 memories it should have found, and 5
questions got no results at all. The three conversations with a person called
John crowded each other most.

**Fix.**
- Filter by user *inside* the search, not after it (a search that supports
  filters, or one index per user).
- Until then, fetch many more results before filtering (for example 200
  instead of 18).
- Measure it: count how many of the 18 results belong to the right user.

**Result — fixed.** The search now looks only at the asking user's memories
and ranks all of them exactly, so other users' data can no longer push them
out. Measured on all questions, compared one by one with the run after fix 2:

| | Before | After |
|---|---|---|
| Right memories found by the search (best 12 per question) | 91.5% | 100% |
| Questions that got no search results | 5 | 0 |
| The 128 most-crowded normal questions answered correctly | 54 | 60 (+6, within luck) |
| Strict score (1,540 questions) | 45.5% | 45.5% (no change) |
| J score (1,540 questions) | 56.8% | 57.1% (+0.3, within luck) |
| Trick (adversarial) questions handled correctly | 66.6% | 61.9% (−4.7, just within luck) |
| Correct answers over all 1,986 questions | 998 | 976 (−22, within luck) |
| Time per question (median) | 19 s | 19 s |

The search is now correct, but on this test it did not change the score: the
memories it used to miss were mostly lower-ranked ones. The fix was kept under
a rule agreed before the run: nothing may get significantly worse. Trick
questions came closest. With more of its own memories, Engram more often
answers with details instead of saying it has no record. About half of the
lost trick questions correct the question ("it was Caroline, not Melanie") but
add details, which the strict judge counts as wrong. Watch this in the next
fixes. An internet outage broke part of the first run, so 1,088 questions
were asked again with the same code and the two parts were combined.

## 6. Messages are stored without their context

**What happens.**
- Each pair of messages is read on its own. The earlier part of the
  conversation is not passed along, so "she", "it" or "that place" cannot
  always be worked out.
- If a name could belong to two known people, the whole message is rejected
  and none of its facts are stored.

**Cost.** Not yet measured. It shows up as answers where Engram "did not
know".

**Fix.**
- Send the previous few turns along with each message (the system already
  accepts this).
- When a name is unclear, skip or flag only that one fact and store the rest.

---

## Order of fixing

1. **Problems 1 and 2 first.** They only change how questions are answered,
   so the stored conversations can be reused — just ask the questions again.
2. **Problems 3 and 6 together.** Both change how conversations are stored,
   so the conversations are stored again once, after both fixes.
3. **Problems 4 and 5.**

Fix one problem at a time and measure each fix on all 1,540 questions,
comparing every question before and after. The answer model is a little
random, so a gain of only 1–2 points can be luck.
