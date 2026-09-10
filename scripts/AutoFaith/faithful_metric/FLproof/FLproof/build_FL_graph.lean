import Lean
import Architect
open Lean Elab Command


structure AutoFaithBlueprintNode where
  name : String
  category : String
  statement : String
  proof : Option String
  directory : String
deriving ToJson

structure AutoFaithBlueprintEdge where
  source : String
  target : String
  kind : String
deriving ToJson

structure AutoFaithBlueprintGraph where
  root : String
  nodes : Array AutoFaithBlueprintNode
  edges : Array AutoFaithBlueprintEdge
deriving ToJson

def autofaithBodyConstants : ConstantInfo -> Array Name
  | .defnInfo value   => value.value.getUsedConstants
  | .thmInfo value    => value.value.getUsedConstants
  | .opaqueInfo value => value.value.getUsedConstants
  | .inductInfo value => value.ctors.toArray
  | _                 => #[]

def autofaithDirectConstants (info : ConstantInfo) : Array Name :=
  info.type.getUsedConstants ++ autofaithBodyConstants info

partial def autofaithDiscover
    (env : Environment)
    (name : Name)
    (depth : Nat)
    (maxDepth : Nat)
    (visited : NameSet := {}) : NameSet :=
  if depth > maxDepth then
    visited
  else if visited.contains name then
    visited
  else
    match env.find? name with
    | none => visited
    | some info =>
        let visited := visited.insert name
        if depth == maxDepth then
          visited
        else
          (autofaithDirectConstants info).foldl
            (fun current dependency =>
              autofaithDiscover env dependency (depth + 1) maxDepth current)
            visited

def autofaithRegisterBlueprint (name : Name) : CoreM Unit := do
  let env <- getEnv

  if (Architect.blueprintExt.find? env name).isSome then
    return

  if !env.contains name then
    return

  let node <- Architect.mkNode name {}

  Architect.blueprintExt.add name node

  modifyEnv fun env =>
    Architect.addLeanNameOfLatexLabel env node.latexLabel name

def autofaithCategory (info : ConstantInfo) : String :=
  match info with
  | .thmInfo _ => "THEOREM"
  | _          => "DEFINITION"

def autofaithProofString (info : ConstantInfo) : Option String :=
  match info with
  | .thmInfo value => some (reprStr value.value)
  | _              => none

def autofaithDirectory (name : Name) : CoreM String := do
  let env <- getEnv
  let moduleName :=
    match env.getModuleIdxFor? name with
    | some index => env.allImportedModuleNames[index]!
    | none       => env.header.mainModule
  return moduleName.toString

def autofaithMakeNode (name : Name) : CoreM AutoFaithBlueprintNode := do
  let info <- getConstInfo name
  let directory <- autofaithDirectory name

  return {
    name := name.toString
    category := autofaithCategory info
    statement := reprStr info.type
    proof := autofaithProofString info
    directory := directory
  }

structure AutoFaithTraversalState where
  visited : NameSet := {}
  nodes : Array AutoFaithBlueprintNode := #[]
  edges : Array AutoFaithBlueprintEdge := #[]

partial def autofaithBuildGraph
    (name : Name)
    (allowed : NameSet)
    (state : AutoFaithTraversalState) :
    CoreM AutoFaithTraversalState := do

  if state.visited.contains name then
    return state

  if !allowed.contains name then
    return state

  let env <- getEnv

  if (Architect.blueprintExt.find? env name).isNone then
    return state

  let node <- autofaithMakeNode name

  let mut state := {
    state with
    visited := state.visited.insert name
    nodes := state.nodes.push node
  }

  let (statementUsed, proofUsed) <- Architect.collectUsed name

  for dependency in statementUsed.toArray do
    if dependency != name && allowed.contains dependency then
      state := {
        state with
        edges := state.edges.push {
          source := name.toString
          target := dependency.toString
          kind := "STATEMENT_USES"
        }
      }
      state <- autofaithBuildGraph dependency allowed state

  for dependency in proofUsed.toArray do
    if dependency != name && allowed.contains dependency then
      state := {
        state with
        edges := state.edges.push {
          source := name.toString
          target := dependency.toString
          kind := "PROOF_USES"
        }
      }
      state <- autofaithBuildGraph dependency allowed state

  return state

syntax (name := autofaith_blueprint_graph)
  "#autofaith_blueprint_graph" ident num : command

@[command_elab autofaith_blueprint_graph]
def elabAutoFaithBlueprintGraph : CommandElab := fun stx => do
  match stx with
  | `(command| #autofaith_blueprint_graph $rootIdent:ident $depthSyntax:num) =>
      let root <-
        liftCoreM <|
          realizeGlobalConstNoOverloadWithInfo rootIdent

      let maxDepth := depthSyntax.getNat
      let env <- getEnv

      let discovered :=
        autofaithDiscover env root 0 maxDepth

      for name in discovered.toArray do
        liftCoreM <| autofaithRegisterBlueprint name

      let state <-
        liftCoreM <|
          autofaithBuildGraph root discovered {}

      let graph : AutoFaithBlueprintGraph := {
        root := root.toString
        nodes := state.nodes
        edges := state.edges
      }

      liftIO <|
        IO.println
          s!"AUTOFAITH_BLUEPRINT_JSON:{(toJson graph).compress}"

  | _ =>
      throwUnsupportedSyntax
