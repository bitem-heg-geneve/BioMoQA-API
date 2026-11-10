import os

# app.config refuses to import without a text source (shared triage guard, §5).
# The pure unit tests never contact it; the stack tests talk to the API only.
os.environ.setdefault("SIBILS_URL", "https://biodiversitypmc.sibils.org/api")
