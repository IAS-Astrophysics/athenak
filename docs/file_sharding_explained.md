# File Sharding Explained

This note explains the three file-layout modes in simple terms:

- one monolithic file for the whole run output
- one file per MPI rank
- one file per node (`single_file_per_node`)

The goal is to describe the computational idea, not every implementation detail.

## The Basic Tradeoff

Every output or restart dump has two jobs:

1. Each MPI rank computes or owns some local data.
2. That data has to be placed on disk in some file layout.

The three modes mostly differ in **how much communication happens before writing** and
**how many files the filesystem has to manage**.

## The Three Modes At A Glance

| Mode | Files per dump | Who coordinates? | Main upside | Main downside |
| --- | ---: | --- | --- | --- |
| Shared / monolithic file | 1 | All ranks in `MPI_COMM_WORLD` | Small file count, simplest final layout | Global contention and global coordination |
| `single_file_per_node` | About `nnodes` | Only ranks on the same node | Good compromise between file count and contention | Some node-local communication is still required |
| `single_file_per_rank` | `nranks` | Nobody beyond the local rank | Simplest write path per rank | Huge file count at scale |

## 1. One Monolithic File

This is the classic "everyone contributes to one big file" approach.

- Every rank computes its local chunk of the data.
- All ranks then cooperate to place their chunk into one shared global file.
- The communication scope is the entire MPI job.

In simple terms: the whole simulation agrees on one global output stream.

This is attractive because the output is easy to reason about and there is only one file
per dump. The problem is that very large jobs can overload the shared file with too many
simultaneous writers or too much global synchronization.

## 2. `single_file_per_rank`

This is the opposite extreme.

- Every rank computes its own local chunk.
- Every rank writes its own file.
- There is essentially no coordination needed to combine rank data before writing.

In simple terms: each rank says "I will dump my own piece and not wait for anyone else."

This is computationally simple and often fast per rank, but it creates a very large number
of files. On huge jobs, the filesystem metadata cost can become the dominant problem.

## 3. `single_file_per_node`

This is the middle ground.

- Every rank still computes its own local chunk of the data.
- Ranks on the **same physical node** are grouped together with a node-local MPI
  communicator (`node_comm`).
- Those ranks cooperate to produce **one file per node**, not one file per rank and not
  one file for the whole job.

In simple terms: the node acts like a small sub-team. Ranks cooperate inside the node, but
they avoid forcing the whole machine to coordinate around one file.

So `single_file_per_node` is not just a filename convention. It is a **two-level strategy**:

- computation stays distributed by rank
- disk layout is grouped by node

That is why it usually scales better than a single global file, while avoiding the file
explosion of per-rank output.

## The Core Computational Idea Behind `single_file_per_node`

The important idea is:

> keep work local to the node whenever possible

That shows up in two ways.

### Write side

For node-sharded output, the code limits coordination to the ranks on one node.

- Each rank prepares the data it owns.
- The ranks on that node determine where each local chunk belongs inside the node's file.
- Depending on the output type, they either:
  - write collectively through `node_comm`, or
  - send/assemble node-local data to one node writer and let that writer emit the file.

So the expensive "who writes where?" problem is solved at node scope, not global scope.

### Read side

For node-sharded restart input, the code also treats the node as the natural unit.

- Restart metadata is read once from the manifest.
- Each node then reads only the node payload shards it needs.
- The current implementation uses collective MPI-IO on `node_comm` for per-node restart
  payloads, so the shard is opened once per node group and read in larger spans instead of
  many tiny random reads.

This keeps the restart traffic more structured and reduces the chance that one restart file
becomes a global hotspot.

## How Restarts Differ Slightly From Other Outputs

For restart files, the current `single_file_per_node` layout is split into:

- one shared manifest file at the top level
- one payload file per node under `node_XXXXXXXX/`

That design exists because restart metadata is global, but the bulk field data is naturally
partitioned by node.

So for restarts, `single_file_per_node` means:

- global information is written once
- heavy payload data is sharded by node

This is more efficient than copying the full global restart manifest into every node shard.

## Why `single_file_per_node` Is Usually The Best Compromise

If the job is very small, all three modes can work well.

At large scale, the usual pattern is:

- monolithic file: too much global coordination
- per-rank file: too many files
- per-node file: a practical balance

`single_file_per_node` reduces the number of files from `nranks` down to about `nnodes`,
which is often a large reduction. At the same time, it avoids forcing every rank in the
whole job to fight over one shared file.

That is the main computational reason this mode exists.

## A Useful Mental Model

Think of the three modes like this:

- shared file: one giant checkout line for the whole cluster
- per-rank file: one checkout line per person
- per-node file: one checkout line per room

`single_file_per_node` works because "one line per room" is often much easier to scale than
either extreme.

## When To Prefer Each Mode

### Shared / monolithic file

Prefer this when:

- the run is modest in size
- you want the simplest possible final file layout
- the filesystem handles shared-file MPI-IO well

### `single_file_per_rank`

Prefer this when:

- simplicity of the rank-local write path matters most
- file count is still manageable
- you are debugging or doing smaller runs

### `single_file_per_node`

Prefer this when:

- the simulation is large enough that one global file is becoming a bottleneck
- per-rank file count is too large
- you want most coordination to stay within a node instead of across the full machine

## What `single_file_per_node` Does Not Change

It does **not** change who owns the physics data during the simulation.

- MeshBlocks and arrays are still distributed by MPI rank.
- Each rank still computes its own local state.
- The mode only changes how that distributed state is gathered or coordinated for I/O.

That is why it is best understood as an **I/O scaling strategy**, not a change to the core
physics decomposition.
