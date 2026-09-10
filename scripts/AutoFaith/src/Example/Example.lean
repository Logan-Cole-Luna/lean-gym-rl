example (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  apply And.intro hp hq
