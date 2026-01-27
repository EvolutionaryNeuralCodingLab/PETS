# Virtual Fitting Guide

**Applies to:** `Virtual_Fitting_Main.f3d` in this repository

## Goal of the Workflow

You will:
- **(A)** Import an animal head mesh (reference only)
- **(B)** Position the left and right camera rigs so each points at its eye
- **(C)** Create TWO lofts connecting the fixed hex base to the left/right loft plates
- **(D)** Export ONE combined "single-print headstage" body as an STL

### What You Should NOT Do

- Remodel the camera rigs, unless you use a different camera
- Edit or rebuild the hex base, unless you have a different anchor interface (the existing hex base component is for a standard 3M hexagonal spacer, but can be remodeled to fit any anchor; this and camera holder remodeling requires some know-how and will not be covered here)
- Move individual camera/mirror bodies directly (move the aim components instead)

## Component Map (What Moves vs What Stays Fixed)

### Fixed / Should Stay Put

- **hex_base:1** - The anchor base; treat as the world reference

### Movable "Rig Handles" (These are the ONLY things you should move)

- **left_aim:1** - Moves camera/mirror/axes for the LEFT eye together
- **right_aim:1** - Moves camera/mirror/axes for the RIGHT eye together

### Helpful Reference Items

- **EyeBall1, EyeBall2** - Visual guides; optional for comfortable alignment

### Typical Contents (Do Not Edit Unless You Know Why)

**left_aim:1** contains bodies like:
- camera_holder
- loft_interface_plate_l
- mirror / mirror holder parts
- construction axes: cam_axis, cam_axis_reflection

**right_aim:1** contains bodies like:
- camera_holder_r
- Loft_interface_plate_r
- construction axes: cam_axis_r, reflection_axis_r

## Step-by-Step Instructions

### 1) Open and Sanity-Check the Design

#### 1.1 Open the Fusion File

- File → Open → select `Virtual_Fitting_Main.f3d`

#### 1.2 In the Browser, Confirm the Following

- You can see: `left_aim:1`, `right_aim:1`, `hex_base:1`
- Only `left_aim:1` and `right_aim:1` should be moved during fitting
- `hex_base:1` should remain fixed in position

#### 1.3 (Optional but Recommended) Create a "Working Copy"

- File → Save As…
- Name it: `Virtual_Fitting_{animalID}_{YYYYMMDD}_fit.f3d`

---

### 2) Import the Animal Mesh (Reference Only)

**Goal:** Bring in your animal model so you can aim the cameras in context.

#### 2.1 Insert the Mesh

- Insert → Insert Mesh
- Choose your mesh file (STL/OBJ)

#### 2.2 Put the Imported Mesh into Its Own Component

- If Fusion creates a MeshBody at the root, right-click it → Create Components from Bodies
- Rename the component to: `animal_mesh_{animalID}`

#### 2.3 Scale the Mesh to Correct Size (Choose ONE Method)

**Method A (Preferred):** Scale using the implanted hex anchor on the animal
- Use the known physical dimension of the hex anchor (across-flats, etc.)
- Measure the same dimension on the mesh (Inspect → Measure)
- Compute scale factor = (real dimension) / (measured mesh dimension)
- Modify → Scale (select the animal mesh component)

**Method B:** Scale using a known anatomical landmark distance
- Example: known distance between two landmarks you can measure on the mesh
- Same workflow: measure → compute scale factor → scale

#### 2.4 (Optional Comfort Step) Place the Reference Eyeballs

Fusion may not carry complex textures well, so use simple geometry as visual guides.
- Position EyeBall1 / EyeBall2 roughly into the eye sockets of the mesh
- This is optional: it is only a visual aid, not required

---

### 3) Aim the Camera Rigs (Move ONLY the Aim Components)

**Goal:** Each camera is oriented toward its respective eye with a sensible inclination angle.

#### 3.1 IMPORTANT RULE

Move only:
- `left_aim:1`
- `right_aim:1`

Do NOT drag individual bodies inside these components.

#### 3.2 Reposition/Rotate Left Camera Rig

- Activate the "Move/Copy" tool (Modify → Move/Copy)
- Selection: choose the COMPONENT `left_aim:1` (not bodies)
- Translate/rotate until the camera's optical path points to the left eye

#### 3.3 Reposition/Rotate Right Camera Rig

- Same process with `right_aim:1`

#### 3.4 Use the Construction Axes as a Precise Aiming Cue

- `left_aim:1` contains "cam_axis" (camera forward axis) and "cam_axis_reflection"
- `right_aim:1` contains "cam_axis_r" and "reflection_axis_r"

#### 3.5 Choose Inclination Angle Carefully

- Avoid extreme angles that will collide with the animal mesh or reduce mirror clearance
- Ensure both rigs have enough physical clearance for printing and mounting
- If needed, hide the animal mesh temporarily to check for rig intersections

---

### 4) Create the Two Lofts (hex_base → Loft Interface Plates)

**Goal:** Create a single printable headstage by joining geometry.

You will do TWO loft operations:
- **Loft #1:** hex_base circular cutoff → left loft interface plate
- **Loft #2:** hex_base circular cutoff → right loft interface plate

#### 4.1 Loft to LEFT Interface Plate

- Create → Loft
- Profiles: select
  - (a) the circular connection cutoff face/edge on `hex_base:1` (left side)
  - (b) the matching profile on `loft_interface_plate_l` (inside `left_aim:1`)
- Operation: set to "Join"
- Confirm: OK

#### 4.2 Loft to RIGHT Interface Plate

- Create → Loft
- Profiles: select
  - (a) the circular connection cutoff face/edge on `hex_base:1` (right side)
  - (b) the matching profile on `Loft_interface_plate_r` (inside `right_aim:1`)
- Operation: set to "Join"
- Confirm: OK

#### 4.3 Verify You Now Have a Single Printable Body

- In the Browser, check Bodies: you should see one combined body (or one per join if not yet merged)
- If you see multiple separate bodies:
  - You likely used "New Body" instead of "Join"
  - Undo and redo the loft(s) with Operation = Join

---

### 5) Export STL for Printing

#### 5.1 Hide Everything That Should NOT Be Printed

Turn off visibility for:
- `animal_mesh_{animalID}`
- EyeBall1 / EyeBall2
- Any non-print helper geometry

#### 5.2 Select Only the Final Printable Body

- Right-click the combined headstage body → Save as Mesh (or 3D Print)

#### 5.3 Export Settings (Typical)

- Format: STL (Binary)
- Refinement: High (or Custom if you need smaller file size)
- Units: confirm match your printer workflow (mm is typical)

#### 5.4 Name the STL Using the Scheme

```
headstage_singlepiece_{animalID}_{YYYYMMDD}.stl
```

---

## Troubleshooting

### A) "When I Move One Camera, Other Stuff Moves Too"

- Make sure you are moving the COMPONENT `left_aim:1` or `right_aim:1` (not a body)
- Quick test: expand `left_aim:1` and `right_aim:1` then select the top component name and move it

### B) "Loft Fails or Produces Twisted Geometry"

- Check you selected the correct matching profiles (left cutoff ↔ left plate; right cutoff ↔ right plate)
- Confirm the profiles are planar and not self-intersecting
- Try selecting edges (profile loops) instead of faces if face selection is ambiguous

### C) "Loft Created a New Body Instead of Joining"

- In Loft dialog: Operation must be "Join"
- If Join is unavailable, you may be lofting to a body that Fusion does not consider intersecting: try extending loft, check profile orientation, or confirm both targets are solid bodies

### D) "My Mesh is Huge/Tiny"

- Your scale factor is likely wrong or units are mismatched
- Re-check the real-world dimension you used (mm vs cm)
- Use Inspect → Measure on the mesh after scaling to confirm

### E) "Exported STL Includes Extra Objects"

- Make sure you selected ONLY the final headstage body when exporting
- Hide everything else before export as a safety measure

---

## Best-Practice Tips for a Robust Workflow

- Keep the animal mesh as reference-only. Do not convert it to BRep unless you must
- Rename your working copy file per animal/date so you can reproduce a fit
- Move ONLY `left_aim:1` and `right_aim:1`. This preserves the internal rig relationships
- After you position rigs, consider creating a Named View ("Left fit", "Right fit") for reproducibility
- If you change anything structural, bump the PATCH version and update this guide alongside the .f3d

---

## Checklist (Quick Validation Before Export)

- [ ] `hex_base:1` has not moved
- [ ] `left_aim:1` points at the left eye; `right_aim:1` points at the right eye
- [ ] Two lofts exist and BOTH are Operation = Join
- [ ] Final result is one printable combined body
- [ ] Animal mesh and helper components are hidden
- [ ] STL exported with correct units and filename
