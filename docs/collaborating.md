# How we work together (without waiting on each other)

This is a plain-language guide for the whole team — no coding needed to follow it.

## What we're building

A system that reads a financial news headline and predicts whether the stock will go **up, down, or neutral** the next day, then writes a short explanation of why. Five people each own one piece, and the pieces pass work down a line:

**Aurora → Nadi → Sabina → Jack → Freddi → Jack (final results)**

## The problem

If each person had to wait for the person before them to finish, we'd be stuck in a queue for weeks. Aurora finishes, then Nadi starts, then Sabina starts… one at a time. That's slow, and a delay from any one person stops everyone.

We want everyone building **at the same time**, starting today.

## The solution: data contracts

Think of a **data contract** as an agreed-upon form. Before anyone starts, we all agree exactly what each handoff looks like — what columns it has, what they're called, and what goes in them.

It's like a restaurant kitchen. The waiter writes the order on a ticket in a format the kitchen already understands. The cook never has to ask "what does this mean?" — the ticket is the contract. The waiter and the cook can both be busy at the same time because they already agreed on what a ticket looks like.

For us, those "tickets" are files (spreadsheets and small data files) that pass from one person to the next. The exact format of every one is written down in **[`data_contracts.md`](./data_contracts.md)**.

## Why this lets us work at the same time

Here's the trick: because we agreed on the format up front, **nobody has to wait for real data to arrive**. We created a folder of realistic *pretend* examples — see [`../mock_data/`](../mock_data/) — one for every handoff.

So Nadi doesn't have to wait for Aurora to finish. Nadi grabs the pretend version of Aurora's output, builds against that, and is ready the moment Aurora's real output shows up — because they'll look identical. Same for everyone down the line. Everyone builds in parallel against the pretend examples.

When the real work connects up, it just fits — like puzzle pieces cut to the same template.

## Who hands what to whom

| Step | Who | Hands over | To |
|---|---|---|---|
| 1 | **Aurora** | the news headlines matched to stock prices, with the right answer labelled | Nadi |
| 2 | **Nadi** | the same data plus her model's guess (up/down/neutral) | Sabina |
| 3 | **Sabina** | a report card: how accurate the guesses were, plus a recommendation | Jack |
| 4 | **Jack** | the decision — good enough, or try again? If "try again," he sends notes back to Nadi | Nadi / onward |
| 5 | **Freddi** | a written explanation for each prediction | Jack |
| 6 | **Jack** | the final combined results everyone reads | the team |

(The exact columns for each are in `data_contracts.md`.)

## The one rule that keeps it working

**Don't change the form on your own.** If you rename a column or change what a file looks like without telling the team, you quietly break it for the person who receives your work — they're expecting the old format. If you need a change, say so, we update the contract and the pretend examples together, and everyone adjusts. Changing the agreement is fine; changing it silently is what hurts us.

## How everyone builds their piece the same way (LangGraph)

We agreed every person builds their piece with the same tool, **LangGraph**.

LangGraph is a tool for building a piece out of small steps with arrows between them — "do this, then this, and if the result looks like X go here, otherwise go there." You draw the steps and the arrows; it runs them in order for you. Aurora's piece, for example, is two steps: *match headlines to prices → label up/down/neutral.* Jack's is a few more, with a fork in the middle (good enough → finish, not good enough → send back to Nadi). Same tool, different steps.

Why agree on one tool: each piece then looks the same from the outside. You hand it the files it needs, it runs its steps, and it saves its own files — and you never have to look inside someone else's piece to use yours. The whole line stays exactly like the tickets above: Aurora's piece leaves its file, Nadi's picks it up, and so on.

**Two things to know:**
- To keep the pieces consistent we added one small shared starter file, `agents/base.py`. It's **new and not final yet** — before you build your piece around it, give it a thumbs-up, same rule as the contract: no surprise changes.
- This is only about *how each piece is built and run*. It does **not** change the files you hand over — those are still set by `data_contracts.md`, and that's still the thing that must not change silently.

Jack's piece (the Manager) already works this way, so there's a finished example to copy. Ask him when you're ready.

## Where to look

- **[`data_contracts.md`](./data_contracts.md)** — the exact format of every handoff.
- **[`../mock_data/`](../mock_data/)** — the realistic pretend examples to build against now.
- **[`architecture.md`](./architecture.md)** — the big-picture diagram of how it all connects.
