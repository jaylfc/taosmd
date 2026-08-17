### Fixed

- The resume pair now has a durable system-crontab arming path (`resume_arm_time.py` prints self-deleting, path-precise crontab lines and a `--fire` mode that removes its own entry). The retry can no longer die with the session, because its arming no longer lives there.
