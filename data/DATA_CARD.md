# Dataset card

## IBM sample

Pinned source: https://github.com/IBM/AMLSim/tree/7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca/sample

The archive `20K_fanin200cycle200.tgz` contains 120,558 transfers between 20,000 accounts.
The upstream metadata states 1,803 fraud-labelled nodes and 200 fan-in / 200 cycle
patterns. `transactions.csv` contains `sourceNodeId,targetNodeId,value,time`.
`nodes.csv` contains `nodeid,isFraud,init_balance,fraudStep`.

**Target construction:** `label = isFraud(sender) OR isFraud(receiver)`.
This creates an account-associated risk proxy (29.73% of transfers), not a claim that
all such transfers are laundering. It can label transfers before a node's `fraudStep`;
that is intentional for this retrospective association task. `fraudStep` is excluded
from predictors and target construction. Do not interpret performance as detection
of the actual onset of illicit activity. A positive account label can encompass both
benign and illicit activity; negative labels need not represent every real-world threat.

There are no missing required values in the evaluated sample. FinGuard rejects invalid
records and unknown node references, retains zero amounts and self-transfers, and
assigns stable IDs from source row positions. Repeated transfers with different IDs
are retained. A source file lacking transaction IDs cannot distinguish an accidentally
repeated row from a legitimate repeated transfer; upstream file hashes fix the input.
No currency or physical time unit is inferred. The amount feature uses `log1p`.

Identity fields are graph keys only, never classifier features. Initial balances and
all target-derived information are excluded. Normal and anomalous training classes
are retained at their natural sample prevalence. This prevalence is far higher than
many operational AML settings; reported precision does not transfer to those settings.

The archive is checksum verified and only named CSV/text members are extracted. The
manifest contains source URL, pinned commit, CSV hashes, exact split counts and versions.
Upstream's original seed/configuration for this archived sample is not established here:
reproduction means rerunning the detection experiment on the exact published sample,
not regenerating the IBM sample byte-for-byte. Do not confuse this sample with IBM's
separate AML transaction datasets or claim new AMLSim simulator execution.

## Demonstration scenario

FinGuard's own deterministic 25-transfer scenario has 20 distinct senders transferring
500 units each to one hub over steps 1–20, followed by five 1,900-unit transfers to
five exits over steps 21–25. Total collected: 10,000; redistributed: 9,500.
These are fictional account IDs. The scenario tests policy behavior and explanation
plumbing; it is not a benchmark and it has no measured real-world false-positive rate.
A legitimate collection/disbursement service could show the same shape.

## Licence and retention

AMLSim repository: Apache-2.0, preserved in `AMLSim-LICENSE` and attributed in `NOTICE`.
No personal banking data are included. Original demonstration data are covered by the
project licence. Generated local SQLite ledgers are excluded from Git.
