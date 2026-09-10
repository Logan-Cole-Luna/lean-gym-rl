type arg = 
  | APPLY of string (* Apply for rewrite and apply tactics*)
  | LET of string * string (* apply for variable declaration or calculation tactic*)
  | UNFOLD of string (* using for unfolding definition and intro variable tactics*)
  | ALGEBRA (* using for algebraic manipulation tactics such as aesop, grind, omega, congr, etc*)
  | UNKNOWN of string (* Using for text when say something trivial but not trivial to proof assistant*)
  | BRANCH of string (* Using branching tactics such as by_cases*)
  | CALC of string (* Using when performing calculation *)
  


type proof_strategy = 
  DIRECT 
| CONTRADICTION 
| CONTRAPOSITIVE 
| INDUCTION 
| UNKNOWN of string

type storage = (string * string * string) list

type proof = proof_strategy * (arg list)

