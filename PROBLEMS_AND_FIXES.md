# Problems and Fixes

This page lists the problems found when Engram's memory was tested on the
LoCoMo question set (1,540 answerable questions about 10 long conversations),
and how to fix each one. The numbers come from that test.

## Summary

| # | Problem | How much it costs | Fix in short |
|---|---|---|---|
| 1 | Found memories are thrown away | 149 questions, all answered wrong | Keep found memories when their label does not match — **fixed** (see result below) |
| 2 | False "conflicting evidence" alarms | 95 questions, all answered wrong | Only call it a conflict for facts that can have one value |
| 3 | Dates stay as "last week" | Date questions: only 24% correct | Turn relative times into real dates when storing |
| 4 | Facts get wrong or vague labels | Feeds problem 1; template answers only 10% correct | A label list for everyday life, fixed synonyms |
| 5 | Other users' memories crowd out search results | Every question (not yet measured) | Filter by user inside the search, not after it |
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
answers Engram builds from these labels are right only 10% of the time.

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
how much *other* data is stored. Not yet measured.

**Fix.**
- Filter by user *inside* the search, not after it (a search that supports
  filters, or one index per user).
- Until then, fetch many more results before filtering (for example 200
  instead of 18).
- Measure it: count how many of the 18 results belong to the right user.

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
