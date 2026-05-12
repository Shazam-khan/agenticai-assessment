# Agentic AI Engineer — Technical Assessment | CONFIDENTIAL

---

**Technical Assessment**

# Agentic AI Engineer

**Team Beauty & Cosmix AI Ventures**

---

| | |
|---|---|
| **Role** | Agentic AI Engineer - founding technical hire |
| **Issued by** | Team Beauty & Cosmix AI Ventures |
| **Duration** | 4-6 hours (do not exceed 7 hours) |
| **Submission** | GitHub repo + Loom walkthrough (10-15 min) + written architecture doc |
| **Deadline** | 72 hours from receipt |
| **LLMs allowed** | Claude API, OpenAI, Groq - your choice, justify it |
| **Tools allowed** | Claude Code, Cursor, Copilot, any AI tooling - encouraged and expected |

---

> *This is not a full-stack engineering test. We are not measuring whether you can build CRUD APIs or scrape websites. We are testing whether you understand multi-agent system design, can reason about failure modes and evaluation, and can architect production-grade AI systems that work reliably at scale. Every task has a trap - the obvious solution scores lower than the thoughtful one.*

---

### What separates this from a standard AI engineering test:

- We expect you to make hard architectural choices and justify them
- Working code is required - but so is the reasoning behind every design decision
- We will stress-test your system with adversarial inputs during evaluation
- The Loom is where we evaluate how you think - narrate your tradeoffs, not just your features
- A coherent 70% submission with great reasoning beats a rushed 100% submission with none

---

## The scenario

You are joining as the founding AI engineer at a new venture being spun out of Team Beauty & Cosmix - a cosmetics manufacturing and private-label operation. The venture will build and sell agentic AI products to other manufacturing businesses. Your first product is an autonomous operations intelligence platform that runs multiple specialised AI agents across a client's business.

This test simulates the core engineering challenges you will face on day one. It has four tasks, each harder than anything in a standard full-stack test. They are designed to be interconnected - the best submissions show evidence of thinking across all four simultaneously.

> *Read all four tasks before writing a single line of code. Spend 15-20 minutes mapping the architecture across all tasks. The cross-task design choices are worth more than any individual task completed in isolation.*

---

## What we are looking for

### Agentic system design
Can you design agents that are reliable, observable, and recoverable? Do you think about tool design, memory architecture, failure handling, and inter-agent communication - or do you just wrap an LLM in a loop?

### Evaluation-first thinking
Can you build systems you can measure? A production AI agent without evals is a liability. We expect to see at least a basic evaluation harness - not just 'it works on my machine'.

### Production awareness
Do you think about latency, cost, token limits, rate limiting, prompt versioning, and graceful degradation? Or do you treat the LLM as a black box that always responds correctly?

### Honest tradeoff documentation
Every architectural decision has a cost. We want to see you name the tradeoffs explicitly - not hide them in working code or omit them from the README. The best submission we have ever received documented seven explicit tradeoffs and explained the upgrade path for each one.

---

## 01 — Multi-agent orchestration system `~ 90 min`

### Context
A manufacturing client needs an AI system that monitors their operations and takes autonomous actions. You will build the orchestration layer - the nervous system that coordinates multiple specialised agents.

### What to build
Design and implement a multi-agent system with the following agents:

- **Supervisor Agent** - receives a natural language objective, breaks it into sub-tasks, delegates to specialist agents, and synthesises results
- **Inventory Agent** - checks raw material stock levels and triggers reorder alerts when stock falls below thresholds
- **Production Agent** - reads active production orders, identifies bottlenecks, and suggests schedule adjustments
- **Report Agent** - compiles outputs from other agents into a structured daily briefing for management

### Hard requirements

1. Agents must communicate via a well-defined message schema - not raw strings passed between functions
2. The Supervisor must be able to retry a failed agent call with a modified instruction before escalating
3. Every agent action must be logged with: timestamp, agent name, input, output, token usage, and latency
4. The system must handle a non-responsive or hallucinating agent without crashing the entire pipeline
5. Implement a simple eval harness: given 5 test scenarios, assert the correct agent was called and the output matches expected schema

### The trap

> *The obvious solution is to chain LLM calls in sequence. The correct solution treats agents as first-class services with defined interfaces, failure modes, and observable state. If your Supervisor is just one big prompt that calls functions - you have built a pipeline, not a multi-agent system. Show us the difference.*

**Deliverable: Working Python implementation. Architecture diagram in your Loom. README must explain: how agents communicate, how failures are handled, and what your eval harness tests and why you chose those 5 scenarios.**

---

## 02 — Long-horizon memory & context management `~ 75 min`

### Context
Your agents need persistent memory across sessions. A customer intake agent that forgets what was discussed last week is useless. A production planning agent that cannot recall decisions made three months ago is dangerous. This task tests whether you can build memory that actually works.

### What to build
Build a memory system for the customer intake agent from Task 1 (or standalone if you prefer) that satisfies all of the following:

- **Episodic memory** - stores raw conversation turns with timestamp, session ID, channel, and participant metadata
- **Semantic memory** - extracts and stores facts learned about a customer ('Lumen Botanicals wants 10k units of serum by Q3') as structured records, not raw text
- **Working memory** - maintains the current session context within the LLM context window without exceeding token limits, using summarisation when needed
- **Memory retrieval** - given a new customer message, retrieves the top-3 most relevant past facts using semantic search, injecting them into the prompt

### Hard requirements

1. The agent must not exceed a configurable token budget per turn (default: 4,000 tokens). Show how you enforce this when memory grows large.
2. Implement memory decay: facts not referenced in 90 days should be downweighted in retrieval but not deleted. Show how you implement this.
3. Demonstrate the memory working across two separate sessions for the same customer - session 2 should reference facts from session 1 without being explicitly told them.
4. Write a test that proves the agent does NOT hallucinate facts that were never stored.

### The trap

> *Stuffing the entire conversation history into the context window is not memory - it is context overflow waiting to happen. A vector database alone is not enough - retrieval without curation produces noise. The correct solution has at least three distinct memory layers with explicit promotion logic between them. Show us how information moves from raw conversation to structured fact to retrievable knowledge.*

**Deliverable: Working implementation with a demo script that runs two sessions for the same customer and shows memory recall in action. Include a diagram of your memory architecture in the README.**

---

## 03 — Tool design, reliability & self-correction `~ 75 min`

### Context
Tools are the hands of an agent. A poorly designed tool is worse than no tool - it produces confident wrong answers. This task tests whether you understand what makes a tool trustworthy, observable, and safe to run autonomously in production.

### What to build
Build a tool library for the Production Agent from Task 1. The library must contain at minimum:

- `get_stock_levels(materials: list[str])` - returns current quantity, unit, reorder level, and days_until_stockout for each material
- `create_purchase_order(material, quantity, supplier_id, urgency)` - creates a PO record and returns confirmation. **THIS TOOL HAS REAL SIDE EFFECTS.**
- `get_production_schedule(date_range: tuple)` - returns active production orders, assigned lines, and completion percentage
- `flag_bottleneck(order_id, reason, severity)` - creates an alert record and notifies the relevant team

### Hard requirements

1. Every tool must validate its inputs before calling any downstream service - reject invalid inputs with a structured error, not an exception
2. `create_purchase_order` must require explicit human-in-the-loop confirmation for orders above a configurable threshold (default: 500 units). Show how you implement this in an agentic context.
3. Every tool call must be idempotent where possible - calling `get_stock_levels` twice must return consistent results; calling `create_purchase_order` twice with the same inputs must NOT create two POs
4. Implement a tool circuit breaker: if a tool fails 3 times in a row, the agent must stop calling it and report the failure to the Supervisor
5. Write tests for at least 3 adversarial inputs per tool: malformed data, boundary conditions, and LLM-generated inputs that look plausible but are wrong

### The trap

> *LLMs confidently pass wrong arguments to tools. A tool that trusts its inputs is a vulnerability. The human-in-the-loop requirement is not just a UI feature - it is a fundamental architectural question: how does an autonomous agent pause, request confirmation, and resume? Showing a working answer to that question is worth more than all five tools combined.*

**Deliverable: Working tool library with a test suite. README must document: the idempotency strategy for each tool, how you implemented human-in-the-loop confirmation, and what your circuit breaker state machine looks like.**

---

## 04 — Evaluation, observability & prompt engineering `~ 60 min`

### Context
An AI system you cannot measure is one you cannot trust. This task tests whether you can build the infrastructure to know when your agents are working correctly - and catch it when they are not.

### Part A - Prompt engineering under constraints

You are given the following badly-written system prompt for the customer intake agent. Rewrite it to be production-grade:

> *"You are a helpful AI assistant. Be friendly and collect customer information. Ask them about their company and what they need. When you have enough information, summarise it. Respond in English or Urdu depending on what language they use."*

Your rewritten prompt must satisfy all of the following - document each explicitly:

1. **Deterministic output schema** - the agent must return structured JSON for lead fields, not free text
2. **Adversarial resistance** - the prompt must not be jailbreakable by a customer who tries to make the agent reveal its instructions or go off-topic
3. **Language commitment** - the agent must commit to the detected language for the entire session, not switch mid-conversation
4. **Graceful field collection** - the agent must collect exactly one missing field per turn, never dump all questions at once
5. **Tool call discipline** - the agent must always call `record_lead_fields` when it learns a value, never accumulate and batch

### Part B - Evaluation harness

Build an automated evaluation harness that scores the intake agent on the following dimensions. Each dimension must have at least 3 test cases:

- **Language fidelity** - responds in the correct language for English, Urdu, and mixed-language inputs
- **Field extraction accuracy** - extracts the correct value for each of the 6 lead fields from varied phrasings
- **Hallucination resistance** - does not invent field values the customer never provided
- **Adversarial robustness** - does not comply with prompt injection attempts embedded in customer messages
- **Conversation naturalness** - does not ask for more than one field per turn (use an LLM-as-judge scorer for this)

### Part C - Observability dashboard

Build a minimal observability layer that captures and displays:

- Per-agent token usage over time (input, output, total, cost estimate)
- Tool call success/failure rates with error categorisation
- Average turns-to-completion for the intake agent
- P50/P95 latency per agent and per tool
- Eval scores over time - so you can detect prompt regressions

> *You do not need a production-grade analytics stack. A SQLite table and a simple HTML dashboard or terminal display is sufficient. What matters is that the data is captured, queryable, and visually surfaced. If we cannot see what your system is doing, we cannot trust it.*

**Deliverable: Rewritten prompt with annotations, eval harness that runs headlessly and prints a score report, and a working observability display showing at least 30 minutes of agent activity.**

---

## Scoring guide

Total: 120 points. Pass threshold: 85+. We score on reasoning depth, not feature count.

| Task | Points | Pass criteria |
|---|---|---|
| Task 1 - Multi-agent orchestration | **/30** | Agents have defined interfaces, failures are handled, eval harness runs, communication schema is explicit |
| Task 2 - Memory architecture | **/25** | Three distinct memory layers, token budget enforced, cross-session recall demonstrated, hallucination test passes |
| Task 3 - Tool design & reliability | **/25** | Input validation present, human-in-the-loop works, idempotency demonstrated, circuit breaker implemented, adversarial tests pass |
| Task 4A - Prompt engineering | **/15** | All 5 constraints satisfied, prompt is jailbreak-resistant, each decision annotated |
| Task 4B - Eval harness | **/10** | All 5 dimensions covered, min 3 cases each, runs headlessly, score report printed |
| Task 4C - Observability | **/5** | All 5 metrics captured and displayed, queryable |
| Architecture coherence | **/5** | Tasks 1-4 share infrastructure, design decisions are consistent across the system, no contradictions |
| Documentation & Loom | **/5** | README explains tradeoffs precisely, Loom narrates reasoning not just features, 'what I would do next' is specific and credible |
| **TOTAL** | **/120** | 85+ = pass. Invite to paid 1-week trial project. |

---

## What separates a good submission from a great one

### Good submission:
- All four tasks attempted with working code
- Agents are separate services with defined inputs and outputs
- Basic error handling and logging present
- README explains what was built

### Great submission:
- The eval harness catches a real failure in the agent - and the candidate explains what caused it and how they fixed it
- The memory system demonstrates actual cross-session recall with evidence (logs, database screenshots, Loom demo)
- The human-in-the-loop implementation works in an agentic context - not just a confirm() dialog bolted on
- The observability dashboard surfaces a genuine insight about the system behaviour during the test run
- The Loom includes at least one moment where the candidate says 'this approach has a problem I did not have time to fix, and here is what I would have done'

> *The best agentic AI engineers we have interviewed spent the first 30 minutes of a test doing nothing but drawing the architecture. They wrote no code until they could answer: what happens when an agent fails? How does the system recover? How do we know it is working correctly? If your Loom does not address all three of those questions, your submission is incomplete regardless of how much code you shipped.*

---

## Submission instructions

- Create a public GitHub repo named: `agenticai-assessment`
- Structure: `/task1`, `/task2`, `/task3`, `/task4` with a `README.md` in each
- Root `README.md` must include: system architecture diagram (ASCII or image), all tradeoffs made, AI tools used and how, what you would build next
- Loom walkthrough: 10-15 minutes. Show each task running. Narrate your architectural decisions - especially the ones you are not happy with
- Architecture document: a 1-2 page written doc (PDF or Markdown) explaining how the four tasks connect as a single system. This is not optional.
- Email: GitHub link + Loom link + architecture doc to the address in your offer correspondence

> **WARNING: We will run your eval harness during evaluation. If it errors or all tests pass trivially (assertions that always pass), the task does not score. Test your tests before submitting.**

> **WARNING: A Loom is mandatory. A submission without a Loom will not be evaluated regardless of code quality. There are no exceptions.**

---

## Questions & clarifications

Make reasonable assumptions and document them. Do not wait for clarification unless you have a genuine technical blocker (e.g. API access). How you handle ambiguity is part of what we are evaluating.

For urgent technical blockers only, contact the email address in your offer correspondence.

---

**Good luck. We are looking for someone who makes us rethink how we build AI systems - not someone who completes a checklist.**

---

*Team Beauty & Cosmix AI Ventures — Confidential*
