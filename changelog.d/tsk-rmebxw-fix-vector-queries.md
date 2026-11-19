### Fixed

- Search strategies named the vector source "qmd" while every caller passes it as "vector", so the fast strategy skipped vector memory for question-shaped queries ("what is ...?", "where does ...?"). The strategies now use "vector".
- The fast strategy always queries the vector store in addition to its primary and secondary sources.
- The minimal strategy still queries one source by design; it falls back to the vector store only when its primary source returns nothing, so a question can still miss on minimal when the primary returns something irrelevant.
