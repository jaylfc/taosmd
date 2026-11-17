### Fixed
- Memory extractor: stop time-word trim producing junk `is_a` facts from lone participles/adverbs/adjectives (e.g. "Content is loaded." no longer yields `is_a(Content, loaded)`, "The server is up now." no longer yields `is_a(server, up)`).
- Memory extractor: anchor meta-text phrases so they match agent boilerplate (whole sentence/line) not phrases inside ordinary sentences (e.g. "Jay manages the loaded below panel." now extracts the manages fact instead of returning empty).
