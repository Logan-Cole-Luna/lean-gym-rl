import Lean
import Architect
import Lean.Elab.DeclarationRange

open Lean
open Lean Meta
open Lean Elab Command

/-!
# AutoFaith source-level declaration metadata

This file intentionally does NOT traverse theorem proof terms or definition
values when constructing graph dependencies.

It has one job:

1. resolve a source-written Lean identifier to its fully qualified declaration;
2. classify the declaration;
3. pretty-print its TYPE / theorem statement;
4. report its defining module and declaration source range;
5. report constants occurring in its TYPE.

The Python side mines the original `.lean` source proof/definition body,
extracts source-written identifiers, asks Lean to resolve those identifiers,
and recursively builds the graph.

Thus proof dependencies come from:

    original source proof
        -> identifier resolution

NOT from:

    ConstantInfo.thmInfo.value.getUsedConstants
-/

structure AutoFaithBlueprintNode where
  name : String
  category : String
  statement : String
  moduleName : String

  sourceStartLine : Option Nat := none
  sourceStartColumn : Option Nat := none
  sourceEndLine : Option Nat := none
  sourceEndColumn : Option Nat := none
deriving ToJson


structure AutoFaithDeclarationInfo where
  query : String
  node : AutoFaithBlueprintNode

  /-
  These are dependencies of the declaration TYPE only.

  We deliberately do not expose dependencies obtained from the theorem proof
  term or definition value.
  -/
  statementUses : Array String
deriving ToJson


/--
AutoFaith-level declaration categories.

`INSTANCE` is checked before `ConstantInfo`, since instances are typically
ordinary definitions/theorems additionally registered for typeclass search.
-/
def autofaithCategory
    (name : Name)
    (info : ConstantInfo) :
    CoreM String := do

  if ← Lean.Meta.isInstance name then
    return "INSTANCE"

  return match info with
  | .thmInfo _ =>
      "THEOREM"

  | .axiomInfo _ =>
      "AXIOM"

  | .inductInfo _ =>
      "INDUCTIVE"

  | .ctorInfo _ =>
      "CONSTRUCTOR"

  | .recInfo _ =>
      "RECURSOR"

  | .defnInfo _ =>
      "DEFINITION"

  | .opaqueInfo _ =>
      "DEFINITION"

  | .quotInfo _ =>
      "DEFINITION"


/-- Return the module in which `name` was introduced. -/
def autofaithModuleName
    (name : Name) :
    CoreM String := do

  let env ← getEnv

  let moduleName :=
    match env.getModuleIdxFor? name with
    | some index =>
        env.allImportedModuleNames[index]!

    | none =>
        env.header.mainModule

  return moduleName.toString


/--
Make one human-readable declaration node.

There is deliberately no `proof` and no `definition` field here.
-/
def autofaithMakeNode
    (name : Name) :
    TermElabM AutoFaithBlueprintNode := do

  let info ← getConstInfo name

  let category ←
    autofaithCategory name info

  let moduleName ←
    autofaithModuleName name

  let statementFmt ←
    Meta.ppExpr info.type

  let ranges? ←
    findDeclarationRanges? name

  let sourceStartLine :=
    ranges?.map fun ranges =>
      ranges.range.pos.line

  let sourceStartColumn :=
    ranges?.map fun ranges =>
      ranges.range.pos.column

  let sourceEndLine :=
    ranges?.map fun ranges =>
      ranges.range.endPos.line

  let sourceEndColumn :=
    ranges?.map fun ranges =>
      ranges.range.endPos.column

  return {
    name := name.toString
    category := category
    statement := statementFmt.pretty
    moduleName := moduleName

    sourceStartLine := sourceStartLine
    sourceStartColumn := sourceStartColumn
    sourceEndLine := sourceEndLine
    sourceEndColumn := sourceEndColumn
  }


/--
Constants directly occurring in the declaration TYPE.

This may contain notation/typeclass infrastructure introduced while
elaborating the statement, but it NEVER reads the theorem proof term or
definition value.
-/
def autofaithStatementUses
    (name : Name) :
    CoreM (Array String) := do

  let info ← getConstInfo name

  return info.type.getUsedConstants.map
    (fun dependency =>
      dependency.toString)


syntax (name := autofaith_decl_info)
  "#autofaith_decl_info" ident : command


/--
Resolve exactly one identifier in Lean's current namespace/open context and
print its metadata.

Python deliberately invokes this command on identifiers that were literally
found in a source proof/definition.  This lets Lean perform name resolution
without using the elaborated theorem proof term as a dependency source.
-/
@[command_elab autofaith_decl_info]
def elabAutoFaithDeclInfo :
    CommandElab := fun stx => do

  match stx with
  | `(command|
      #autofaith_decl_info $queryIdent:ident) =>

      let resolved ←
        liftCoreM <|
          realizeGlobalConstNoOverloadWithInfo
            queryIdent

      let node ←
        liftTermElabM <|
          autofaithMakeNode
            resolved

      let statementUses ←
        liftCoreM <|
          autofaithStatementUses
            resolved

      let result : AutoFaithDeclarationInfo := {
        query := queryIdent.getId.toString
        node := node
        statementUses := statementUses
      }

      liftIO <|
        IO.println
          s!"AUTOFAITH_DECL_JSON:{(toJson result).compress}"

  | _ =>
      throwUnsupportedSyntax
