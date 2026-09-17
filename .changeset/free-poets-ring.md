---
'@mastra/memory': patch
---

Fixed deleted threads leaving observational memory behind. Deleting a thread now waits for any in-flight observational-memory cycle on that thread before cleaning up, and a cycle that finishes after its thread was deleted no longer writes observation vectors or recreates the thread's memory record. Resource-scoped memory is covered too: a cycle that outlives a deleted thread no longer adds that thread's observations to the shared resource record, where they were previously served to every other thread of the resource. Previously, text from a deleted thread could stay searchable through resource-scoped recall. Fixes #23177.
