"""The wall clock a sandbox shell reads, in milliseconds, whatever `date` the image ships.

`date +%s%3N` is GNU-only. uutils coreutils (Ubuntu 25.10 images, eclipse-temurin among them)
ignores the width and prints all nine digits of the nanoseconds, so a 16 ms run would be
reported as 16,344,562 ms; busybox prints the literal
`%N`. The snippet below asks for `%s%N`, keeps the digits, pads with three zeros, and takes
the first thirteen: nanoseconds (19 digits) and milliseconds (13) both come out as
milliseconds, and a `date` with no `%N` at all gives seconds x 1000. Every tool it needs
(date, tr, printf, cut) is in coreutils and busybox alike.
"""

from __future__ import annotations

# Shell text of a command substitution's body: use as  __t0=$(NOW_MS_SH)
NOW_MS_SH = "printf '%s000' \"$(date +%s%N | tr -cd 0-9)\" | cut -c1-13"
