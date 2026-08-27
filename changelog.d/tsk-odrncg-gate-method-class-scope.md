### Fixed
- The duplicate-definition gate no longer false-positives on a class defined inside a method: the class-scope guard now distinguishes a method closure scope (`class Outer > make`) from a bare class scope (`class Foo`), so a class-in-a-method does not collide with a same-named module-level class.
