#!/usr/bin/env python3
"""
Isaac Sim UAV Scene Generator

Creates a simple USD scene with a UAV, camera, and target for
testing the Lagari vision system.

Usage:
    ./isaac_sim.sh --exec scripts/isaac_sim/create_uav_scene.py -- \
        --output /path/to/scene.usd
"""

import argparse
from pathlib import Path


def create_scene():
    """Create a sample UAV scene for testing"""
    import omni
    from pxr import Usd, UsdGeom, Gf, UsdPhysics, UsdLux
    from omni.isaac.core import World
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.stage import create_new_stage
    
    # Create a new stage
    create_new_stage()
    stage = omni.usd.get_context().get_stage()
    
    # Set up axis and units
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    
    # Create root xform
    world_prim = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world_prim.GetPrim())
    
    # Add ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr([
        (-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)
    ])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateNormalsAttr([(0, 0, 1)] * 4)
    ground.GetPrim().GetAttribute("primvars:displayColor").Set([(0.3, 0.5, 0.3)])
    
    # Add physics collision
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    
    # Create UAV body (simplified box representation)
    uav_xform = UsdGeom.Xform.Define(stage, "/World/UAV")
    uav_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 10))  # 10m altitude
    
    uav_body = UsdGeom.Cube.Define(stage, "/World/UAV/Body")
    uav_body.CreateSizeAttr(0.3)
    uav_body.CreateDisplayColorAttr([(0.2, 0.2, 0.8)])
    
    # Create camera attached to UAV (downward-facing)
    camera = UsdGeom.Camera.Define(stage, "/World/UAV/Camera")
    camera_xform = camera.GetPrim().GetAttribute("xformOp:transform")
    if not camera_xform:
        camera.AddTransformOp()
    
    # Rotate camera to face down (Z-axis pointing down)
    camera.AddRotateXYZOp().Set(Gf.Vec3f(180, 0, 0))
    camera.AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.2))
    
    # Camera properties
    camera.CreateFocalLengthAttr(18.0)  # Wide-angle for UAV
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
    
    # Create target (landing pad with H pattern)
    target_xform = UsdGeom.Xform.Define(stage, "/World/Target")
    target_xform.AddTranslateOp().Set(Gf.Vec3f(5, 3, 0.01))  # Offset from center
    
    # Landing pad base (circle approximated by cylinder)
    pad_base = UsdGeom.Cylinder.Define(stage, "/World/Target/Pad")
    pad_base.CreateRadiusAttr(1.0)
    pad_base.CreateHeightAttr(0.02)
    pad_base.CreateDisplayColorAttr([(0.8, 0.8, 0.8)])
    
    # Add H marker (simplified as two cubes)
    h_left = UsdGeom.Cube.Define(stage, "/World/Target/H_Left")
    h_left.CreateSizeAttr(0.3)
    h_left.AddTranslateOp().Set(Gf.Vec3f(-0.3, 0, 0.15))
    h_left.AddScaleOp().Set(Gf.Vec3f(0.2, 1.0, 0.05))
    h_left.CreateDisplayColorAttr([(1.0, 0.5, 0.0)])
    
    h_right = UsdGeom.Cube.Define(stage, "/World/Target/H_Right")
    h_right.CreateSizeAttr(0.3)
    h_right.AddTranslateOp().Set(Gf.Vec3f(0.3, 0, 0.15))
    h_right.AddScaleOp().Set(Gf.Vec3f(0.2, 1.0, 0.05))
    h_right.CreateDisplayColorAttr([(1.0, 0.5, 0.0)])
    
    h_cross = UsdGeom.Cube.Define(stage, "/World/Target/H_Cross")
    h_cross.CreateSizeAttr(0.3)
    h_cross.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.15))
    h_cross.AddScaleOp().Set(Gf.Vec3f(1.0, 0.2, 0.05))
    h_cross.CreateDisplayColorAttr([(1.0, 0.5, 0.0)])
    
    # Add lighting
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1000)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))
    
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr(500)
    dome.CreateTextureFileAttr("omniverse://localhost/NVIDIA/Assets/Skies/Cloudy/cloudy.hdr")
    
    return stage


def save_scene(output_path: str):
    """Save the created scene to a USD file"""
    import omni
    stage = omni.usd.get_context().get_stage()
    stage.Export(output_path)
    print(f"Scene saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create Isaac Sim UAV scene")
    parser.add_argument("--output", "-o", 
                        default="./scenes/uav_test_scene.usd",
                        help="Output USD file path")
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create and save scene
    create_scene()
    save_scene(str(output_path))


if __name__ == "__main__":
    main()
