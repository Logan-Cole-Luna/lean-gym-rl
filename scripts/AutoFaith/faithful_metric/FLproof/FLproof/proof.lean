import Mathlib
import Architect

-- Your AutoFaith / Blueprint graph extraction code above ...
import FLproof.BlueprintGraph

theorem main (a b : ℤ) (ha : 2 ∣ a) (hb : 2 ∣ b) :
    2 ∣ a + b := by
  rcases ha with ⟨k, hk⟩
  rcases hb with ⟨l, hl⟩
  use k + l
  calc
  a + b = 2 * k + 2 * l := by rw [hk, hl]
  _ = 2 * (k + l) := by ring

#autofaith_decl_info main
#check HAdd
#check Monoid ℝ
