# Renderer Adapter

The 3D shoe in the `skatingbergs` project is only an example consumer of the display quaternion. The reusable contract is:

```text
displayQuaternion -> renderer adapter -> object transform
```

Do not tie BLE parsing, estimator state, or pose calibration to a specific mesh.

## Required Abstraction

Use a renderer-facing interface equivalent to:

```kotlin
interface RendererAdapter {
    fun render(displayQuaternion: Quaternionf)
}
```

Everything upstream should remain unaware of:

- mesh topology
- engine-specific coordinate systems
- scene graph ownership
- camera behavior
- materials or shading

## What Must Stay Stable

- The renderer consumes the already-calibrated `displayQuaternion`.
- The estimator produces `absoluteQuaternion`.
- Pose calibration converts absolute to display orientation.
- The object type can change without changing the sensor pipeline.

## Adapting To Common Engines

### OpenGL ES / Filament / low-level native rendering

- Convert quaternion to a rotation matrix.
- Compose the model transform from the rotation matrix plus the model's translation and scale.
- Keep the object-origin correction separate from the sensor quaternion.

### Unity

- Convert the quaternion into Unity's handedness and axis convention.
- Apply a fixed model-space correction if the mesh forward/up axes differ from the sensor/object convention.
- Keep the sensor-to-display quaternion separate from the mesh import correction.

### SceneKit / RealityKit

- Convert the display quaternion into the framework's quaternion type.
- Apply fixed scene correction in a parent transform or separate node if needed.

### Three.js / WebGL

- Convert the quaternion into Three.js order and handedness expectations.
- Keep sensor quaternion and mesh correction as separate transforms.

## Recommended Integration Pattern

Use two transforms:

1. `displayQuaternion`: live orientation from the sensor pipeline
2. `modelCorrection`: fixed per-object correction for mesh import orientation

Final object transform:

```text
finalObjectRotation = displayQuaternion * modelCorrection
```

If the object seems rotated incorrectly when the sensor is otherwise correct, adjust `modelCorrection`, not the sensor parser or estimator.

## Acceptance Examples

- Replacing the skating shoe with a cube should not change the sensor math.
- Replacing the shoe with a racket, avatar, or camera rig should only require renderer-side adaptation.
- If the app shows the same debug Euler/quaternion values but the object looks wrong, the bug is in the renderer adapter or mesh correction, not the sensor pipeline.
