Write-Ahead Logging - Checkpointing
===================================

#### Checkpoint process

1.  **Redo log** is the WAL ﬁle storing committed transactions.
2.  **Checkpoint** copies committed transactions from WAL back into the database.
3.  **Truncate** resets WAL size in *truncate* mode.
4.  **Restart** reuses WAL in *restart* mode.

!

During normal operation, all reads go to the main database, but writes append to the WAL. The WAL periodically must be checkpointed back into the database.

### `PRAGMA wal_checkpoint`

```
PRAGMA wal_checkpoint;
```

Performs a checkpoint operation.  Options `TRUNCATE`, `FULL`, `RESTART`, or `PASSIVE` can be supplied.  `PASSIVE` flushes without blocking writers.  `FULL` waits for writers to finish.  `RESTART` waits and then resets the WAL so new writes start a new snapshot.  `TRUNCATE` does restart + truncate the `-wal` ﬁle.

#### Automatic checkpoints

SQLite automatically schedules checkpoints when the WAL grows beyond a threshold controlled by `PRAGMA wal_autocheckpoint`.  Default is 1000 pages; higher values reduce contention at cost of larger WAL ﬁles.

#### Practical considerations

* Frequent checkpoints keep the WAL small but can reduce concurrency.
* Less frequent checkpoints increase WAL size, potentially affecting disk usage and recovery cost.
* Background checkpointing avoids pausing writers; use an external thread to invoke `sqlite3_wal_checkpoint_v2()`.

#### Monitoring checkpoints

* `PRAGMA wal_checkpoint(PASSIVE);`
* `sqlite3_wal_checkpoint_v2(db, NULL, SQLITE_CHECKPOINT_TRUNCATE, &nLog, &nCkpt);`

This reports pages in the WAL and pages moved to the database.
