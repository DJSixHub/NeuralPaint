from __future__ import annotations

import json
import math
import random
import hashlib
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import bmesh
import bpy
import bpy_extras
from mathutils import Euler, Vector

# Output configuration
NUM_SAMPLES = 5
RESOLUTION_X = 1280
RESOLUTION_Y = 720
OUTPUT_ROOT = Path(r"E:\Escuela\Redes Neuronales\NeuralPaint\dataset_sintetico")
COLOR_DIR = OUTPUT_ROOT / "images"
ANNOTATION_DIR = OUTPUT_ROOT / "annotations"
MASK_DIR = OUTPUT_ROOT / "masks"

# Assets
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
WALLS_DIR = DATASET_DIR / "walls"
FLOORS_DIR = DATASET_DIR / "floors"
DIRECT_LOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".exr", ".hdr", ".dds", ".tga"}
CONVERTIBLE_EXTENSIONS = {".webp", ".avif"}
IMAGE_EXTENSIONS = DIRECT_LOAD_EXTENSIONS | CONVERTIBLE_EXTENSIONS
PROJECTION_IMAGES: List[Path] = [
    p for p in DATASET_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]
WALL_TEXTURES: List[Path] = [
    p for p in WALLS_DIR.glob("**/*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]
FLOOR_TEXTURES: List[Path] = [
    p for p in FLOORS_DIR.glob("**/*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]
TEXTURE_CACHE_DIR = BASE_DIR / "_converted_textures"
TEXTURE_CONVERSION_CACHE: dict[Path, Path] = {}

if not WALL_TEXTURES:
    print("[Aviso] No se encontraron texturas en dataset/walls; se usará un material procedimental para las paredes.")
if not FLOOR_TEXTURES:
    print("[Aviso] No se encontraron texturas en dataset/floors; se usará un material procedimental para el suelo.")
if not PROJECTION_IMAGES:
    raise RuntimeError("Place at least one projection image inside dataset/")


def build_texture_cycle(paths: Sequence[Path]) -> Deque[Path]:
    if not paths:
        raise ValueError("Texture list cannot be empty")
    return deque(paths)


def next_texture(queue: Deque[Path]) -> Path:
    if not queue:
        raise ValueError("Texture queue is empty")
    texture = queue.popleft()
    queue.append(texture)
    return texture


def _converted_texture_path(original: Path) -> Path:
    digest = hashlib.sha1(str(original).encode("utf-8")).hexdigest()[:12]
    return TEXTURE_CACHE_DIR / f"{original.stem}_{digest}.png"


def ensure_texture_compatible(path: Path) -> Path:
    suffix = path.suffix.lower()
    if not path.exists():
        raise RuntimeError(f"Textura no encontrada: {path}")
    if suffix in DIRECT_LOAD_EXTENSIONS:
        return path
    if suffix not in CONVERTIBLE_EXTENSIONS:
        raise RuntimeError(f"Formato de textura no soportado: {path.suffix}")

    cached = TEXTURE_CONVERSION_CACHE.get(path)
    if cached and cached.exists() and cached.stat().st_mtime >= path.stat().st_mtime:
        return cached

    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Se requiere Pillow para convertir texturas .webp/.avif a PNG. "
            "Instala con 'pip install Pillow'."
        ) from exc

    converted_path = _converted_texture_path(path)
    converted_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(path) as img:
            img = img.convert("RGBA")
            img.save(converted_path, format="PNG")
    except Exception as exc:  # pragma: no cover - image conversion runtime
        raise RuntimeError(f"No se pudo convertir la textura {path.name}: {exc}") from exc

    TEXTURE_CONVERSION_CACHE[path] = converted_path
    return converted_path


def load_texture_image(path: Path) -> bpy.types.Image:
    compatible_path = ensure_texture_compatible(path)
    try:
        return bpy.data.images.load(str(compatible_path), check_existing=True)
    except RuntimeError as exc:
        raise RuntimeError(f"No se pudo cargar la textura {path.name}: {exc}") from exc


def configure_cycles(use_gpu: bool = True) -> None:
    bpy.context.scene.render.engine = "CYCLES"
    scene = bpy.context.scene
    scene.cycles.samples = random.randint(96, 192)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 6
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 6
    scene.cycles.volume_bounces = 2
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_motion_blur = False
    if use_gpu:
        gpu_configured = False
        try:
            preferences = bpy.context.preferences
            if "cycles" not in preferences.addons:
                bpy.ops.preferences.addon_enable(module="cycles")
            prefs = preferences.addons["cycles"].preferences
            for backend in ("OPTIX", "CUDA", "HIP", "METAL"):
                try:
                    prefs.compute_device_type = backend
                except Exception:
                    continue
                devices = prefs.get_devices() or []
                for device_group in devices:
                    if device_group is None:
                        continue
                    for device in device_group:
                        try:
                            device.use = True
                        except AttributeError:
                            continue
                        if getattr(device, "type", "CPU") != "CPU":
                            gpu_configured = True
                if gpu_configured:
                    break
            if gpu_configured:
                scene.cycles.device = "GPU"
            else:
                print("[Aviso] No se detectó GPU compatible para Cycles; se usará render CPU.")
                scene.cycles.device = "CPU"
        except Exception:
            print("[Aviso] Falló la configuración de GPU para Cycles; se usará render CPU.")
            scene.cycles.device = "CPU"
    else:
        scene.cycles.device = "CPU"

    view_settings = scene.view_settings
    view_settings.exposure = random.uniform(-0.35, 0.45)
    view_settings.gamma = random.uniform(0.95, 1.05)


def clean_scene(remove_data_blocks: bool = True) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    if remove_data_blocks:
        for datablock in (
            bpy.data.meshes,
            bpy.data.materials,
            bpy.data.lights,
            bpy.data.cameras,
            bpy.data.curves,
        ):
            for block in list(datablock):
                datablock.remove(block)


def add_room(width: float, depth: float, height: float, wall_texture: Optional[Path]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1)
    room = bpy.context.active_object
    room.name = "RoomShell"
    room.scale = (width / 2.0, depth / 2.0, height / 2.0)
    bpy.ops.object.transform_apply(scale=True)
    bpy.context.view_layer.objects.active = room
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=89.0, island_margin=0.03)
    bpy.ops.object.mode_set(mode="OBJECT")

    mat = bpy.data.materials.new("RoomMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    for name, value in (
        ("Specular", 0.15),
        ("Specular Tint", 0.2),
        ("Roughness", 0.7),
    ):
        socket = principled.inputs.get(name)
        if socket is None:
            continue
        current = socket.default_value
        if isinstance(current, float):
            socket.default_value = value
        elif isinstance(current, (tuple, list)):
            length = len(current)
            if length == 4:
                socket.default_value = (value, value, value, current[-1])
            elif length == 3:
                socket.default_value = (value, value, value)

    if wall_texture is not None:
        tex_node = nodes.new(type="ShaderNodeTexImage")
        tex_node.image = load_texture_image(wall_texture)
        tex_node.interpolation = "Cubic"

        tex_coord = nodes.new(type="ShaderNodeTexCoord")
        mapping = nodes.new(type="ShaderNodeMapping")
        links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex_node.inputs["Vector"])
        mapping.inputs["Scale"].default_value = (
            random.uniform(0.6, 1.4),
            random.uniform(0.6, 1.4),
            1.0,
        )
        mapping.inputs["Rotation"].default_value = (
            0.0,
            0.0,
            random.uniform(0, math.tau),
        )

        links.new(tex_node.outputs["Color"], principled.inputs["Base Color"])
    else:
        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = random.uniform(1.5, 4.0)
        noise.inputs["Detail"].default_value = random.uniform(6.0, 12.0)
        ramp = nodes.new(type="ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (
            random.uniform(0.4, 0.6),
            random.uniform(0.4, 0.6),
            random.uniform(0.4, 0.6),
            1,
        )
        ramp.color_ramp.elements[1].color = (
            random.uniform(0.15, 0.3),
            random.uniform(0.15, 0.3),
            random.uniform(0.15, 0.3),
            1,
        )
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], principled.inputs["Base Color"])

    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    room.data.materials.append(mat)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.normals_make_consistent(inside=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    return room


def add_floor_material(room: bpy.types.Object, texture_path: Optional[Path]) -> None:
    mat = bpy.data.materials.new("FloorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = random.uniform(0.25, 0.55)
    principled.inputs["Specular"].default_value = random.uniform(0.05, 0.12)

    if texture_path is not None:
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.image = load_texture_image(texture_path)
        tex.interpolation = "Cubic"
        tex.extension = "REPEAT"
        tex_coord = nodes.new(type="ShaderNodeTexCoord")
        mapping = nodes.new(type="ShaderNodeMapping")
        links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
        scale_factor = random.uniform(0.8, 2.5)
        mapping.inputs["Scale"].default_value = (scale_factor, scale_factor, 1.0)
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, random.uniform(0, math.tau))
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], principled.inputs["Base Color"])
    else:
        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = random.uniform(3, 8)
        noise.inputs["Detail"].default_value = 12
        color_ramp = nodes.new(type="ShaderNodeValToRGB")
        color_ramp.color_ramp.elements[0].color = (
            random.uniform(0.08, 0.18),
            random.uniform(0.08, 0.18),
            random.uniform(0.08, 0.18),
            1,
        )
        color_ramp.color_ramp.elements[1].color = (
            random.uniform(0.3, 0.5),
            random.uniform(0.3, 0.5),
            random.uniform(0.3, 0.5),
            1,
        )
        links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
        links.new(color_ramp.outputs["Color"], principled.inputs["Base Color"])
        bump = nodes.new(type="ShaderNodeBump")
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        bump.inputs["Strength"].default_value = 0.18
        links.new(bump.outputs["Normal"], principled.inputs["Normal"])

    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    room.data.materials.append(mat)
    bpy.context.view_layer.objects.active = room
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(room.data)
    floor_face = min(bm.faces, key=lambda f: sum(v.co.z for v in f.verts) / len(f.verts))
    floor_face.material_index = len(room.data.materials) - 1
    bmesh.update_edit_mesh(room.data)
    bpy.ops.object.mode_set(mode="OBJECT")


def add_projector_plane(width: float, height: float, image_path: Path) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=1)
    plane = bpy.context.active_object
    plane.name = "ProjectionPlane"
    plane.scale = (width / 2.0, height / 2.0, 1)
    plane.rotation_euler = Euler((math.radians(90), 0, 0))
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.new("ProjectionMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.92, 0.92, 0.92, 1.0)
    principled.inputs["Roughness"].default_value = random.uniform(0.15, 0.35)
    principled.inputs["Specular"].default_value = 0.05

    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    plane.data.materials.append(mat)
    plane["projection_image"] = image_path.name
    return plane


def add_projector_light(plane: bpy.types.Object, room_depth: float, image_path: Path) -> bpy.types.Object:
    bpy.ops.object.light_add(type="SPOT")
    lamp = bpy.context.active_object
    lamp.name = "ProjectorLight"
    lamp.data.use_nodes = True
    lamp.data.use_shadow = True
    lamp.data.shadow_soft_size = random.uniform(0.06, 0.2)

    offset = Vector((
        random.uniform(-0.5, 0.5),
        random.uniform(1.4, 3.2),
        random.uniform(0.8, 1.6),
    ))
    lamp.location = plane.location + offset
    lamp.location.y = min(lamp.location.y, room_depth / 2 - 0.4)
    direction = (plane.location - lamp.location).normalized()
    lamp.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()

    nodes = lamp.data.node_tree.nodes
    links = lamp.data.node_tree.links
    nodes.clear()

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    image_node = nodes.new(type="ShaderNodeTexImage")
    image_node.image = load_texture_image(image_path)
    image_node.image.colorspace_settings.name = "sRGB"
    image_node.interpolation = "Cubic"
    image_node.extension = "CLIP"
    image_node.projection = 'FLAT'
    links.new(mapping.outputs["Vector"], image_node.inputs["Vector"])

    emission = nodes.new(type="ShaderNodeEmission")
    links.new(image_node.outputs["Color"], emission.inputs["Color"])

    light_output = nodes.new(type="ShaderNodeOutputLight")
    links.new(emission.outputs["Emission"], light_output.inputs["Surface"])

    distance = (plane.location - lamp.location).length
    projection_extent = max(plane.dimensions.x, plane.dimensions.z)
    angle = 2 * math.atan((projection_extent * 0.6) / max(distance, 1e-3))
    lamp.data.spot_size = min(max(angle, math.radians(15)), math.radians(120))
    lamp.data.spot_blend = random.uniform(0.02, 0.2)

    mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    mapping.inputs["Rotation"].default_value = (math.radians(90), 0.0, 0.0)
    mapping.inputs["Translation"].default_value = (0.5, 0.5, 0.0)

    projector_energy = random.uniform(900, 6500)
    lamp.data.energy = projector_energy
    emission.inputs["Strength"].default_value = random.uniform(25.0, 95.0)
    tint_choice = random.choice(["neutral", "warm", "cool"])
    if tint_choice == "warm":
        lamp.data.color = (
            random.uniform(0.9, 1.0),
            random.uniform(0.7, 0.9),
            random.uniform(0.6, 0.85),
        )
    elif tint_choice == "cool":
        lamp.data.color = (
            random.uniform(0.6, 0.85),
            random.uniform(0.75, 0.95),
            random.uniform(0.9, 1.0),
        )
    else:
        lamp.data.color = (
            random.uniform(0.85, 1.0),
            random.uniform(0.85, 1.0),
            random.uniform(0.85, 1.0),
        )

    lamp["projection_energy"] = projector_energy
    lamp["projection_image"] = image_path.name
    lamp["projection_tint"] = tint_choice
    return lamp


def add_environment_lights(room_dims: Tuple[float, float, float]) -> dict:
    info: dict = {}
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    strength = random.uniform(0.05, 2.2)
    bg = nodes.get("Background")
    if bg:
        color = (
            random.uniform(0.3, 0.8),
            random.uniform(0.3, 0.8),
            random.uniform(0.3, 0.8),
            1,
        )
        bg.inputs["Strength"].default_value = strength
        bg.inputs["Color"].default_value = color
        info["environment_strength"] = round(strength, 3)
        info["environment_color"] = [round(c, 4) for c in color[:3]]

    scenario = random.choice(["ceiling", "window", "sun"])
    info["lighting_scenario"] = scenario

    if scenario == "ceiling":
        bpy.ops.object.light_add(type="AREA")
        area = bpy.context.active_object
        area.location = (0, 0, room_dims[2] - 0.05)
        area.data.shape = 'RECTANGLE'
        area.data.size = room_dims[0] * random.uniform(0.4, 0.9)
        area.data.size_y = room_dims[1] * random.uniform(0.4, 0.9)
        area.data.energy = random.uniform(220, 650)
        area.data.color = (
            random.uniform(0.85, 1.0),
            random.uniform(0.85, 1.0),
            random.uniform(0.8, 0.95),
        )
        info["area_light_energy"] = round(area.data.energy, 2)
    elif scenario == "window":
        bpy.ops.object.light_add(type="AREA")
        area = bpy.context.active_object
        side = random.choice([-1, 1])
        area.location = (room_dims[0] * 0.45 * side, -room_dims[1] * 0.1, room_dims[2] * random.uniform(0.4, 0.75))
        area.rotation_euler = Euler((0, math.radians(90), math.radians(90 if side > 0 else -90)))
        area.data.shape = 'RECTANGLE'
        area.data.size = room_dims[2] * random.uniform(0.6, 1.0)
        area.data.size_y = room_dims[0] * random.uniform(0.15, 0.25)
        area.data.energy = random.uniform(150, 420)
        area.data.color = (
            random.uniform(0.9, 1.0),
            random.uniform(0.95, 1.0),
            random.uniform(0.9, 1.0),
        )
        info["window_light_energy"] = round(area.data.energy, 2)
    else:
        bpy.ops.object.light_add(type="SUN")
        sun = bpy.context.active_object
        sun.location = (
            random.uniform(-room_dims[0], room_dims[0]),
            random.uniform(-room_dims[1], room_dims[1]),
            room_dims[2] + random.uniform(0.5, 2.0),
        )
        sun.rotation_euler = Euler((random.uniform(math.radians(15), math.radians(60)), 0, random.uniform(-math.pi, math.pi)))
        sun.data.energy = random.uniform(1.5, 4.0)
        sun.data.angle = random.uniform(math.radians(1.0), math.radians(5.0))
        info["sun_energy"] = round(sun.data.energy, 3)
        bpy.ops.object.light_add(type="AREA")
        fill = bpy.context.active_object
        fill.location = (0, 0, random.uniform(0.6, room_dims[2] * 0.6))
        fill.data.shape = 'SQUARE'
        fill.data.size = random.uniform(0.6, 1.4)
        fill.data.energy = random.uniform(80, 220)
        info["fill_light_energy"] = round(fill.data.energy, 2)
    return info


def add_obstacles(plane: bpy.types.Object, count: int) -> List[bpy.types.Object]:
    obstacles: List[bpy.types.Object] = []
    for _ in range(count):
        shape_type = random.choice(["CUBE", "CYLINDER", "SPHERE"])
        if shape_type == "CUBE":
            bpy.ops.mesh.primitive_cube_add(size=1)
        elif shape_type == "CYLINDER":
            bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, segments=24, ring_count=16)
        obj = bpy.context.active_object
        obj.name = f"Obstacle_{shape_type}"
        obj.scale = (
            random.uniform(0.1, 0.6),
            random.uniform(0.1, 0.6),
            random.uniform(0.3, 1.2),
        )
        obj.location = plane.location + Vector((
            random.uniform(-0.5, 0.5),
            random.uniform(0.2, 1.2),
            random.uniform(-0.3, 0.6),
        ))
        obj.rotation_euler = Euler((
            0.0,
            random.uniform(-0.1, 0.1),
            random.uniform(0, math.tau),
        ))
        make_random_principled(obj)
        obj["shape"] = shape_type
        obstacles.append(obj)
    return obstacles


def make_random_principled(obj: bpy.types.Object) -> None:
    mat = bpy.data.materials.new(f"Mat_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (
        random.uniform(0.1, 0.9),
        random.uniform(0.1, 0.9),
        random.uniform(0.1, 0.9),
        1,
    )
    principled.inputs["Roughness"].default_value = random.uniform(0.2, 0.8)
    principled.inputs["Metallic"].default_value = random.uniform(0.0, 0.4)
    obj.data.materials.append(mat)


def setup_camera(plane: bpy.types.Object, room_dims: Tuple[float, float, float]) -> bpy.types.Object:
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.data.lens = random.uniform(28, 45)

    yaw = random.uniform(-math.pi / 6, math.pi / 6)
    radius = random.uniform(1.6, 4.5)
    cam_offset = Vector((
        radius * math.sin(yaw),
        radius * math.cos(yaw)
    ))
    x_pos = plane.location.x + cam_offset.x
    x_pos = max(-room_dims[0] / 2 + 0.2, min(room_dims[0] / 2 - 0.2, x_pos))
    y_pos = plane.location.y + cam_offset.y
    y_pos = max(plane.location.y + 0.4, min(room_dims[1] / 2 - 0.2, y_pos))
    z_upper = min(room_dims[2] - 0.2, plane.location.z + 1.4)
    z_lower = max(0.6, plane.location.z - 0.6)
    if z_upper <= z_lower:
        z_upper = z_lower + 0.2
    camera.location = Vector((
        x_pos,
        y_pos,
        random.uniform(z_lower, z_upper)
    ))
    camera_constraint = camera.constraints.new(type="TRACK_TO")
    camera_constraint.target = plane
    camera_constraint.track_axis = "TRACK_NEGATIVE_Z"
    camera_constraint.up_axis = "UP_Y"

    bpy.context.scene.camera = camera
    return camera


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def project_vertices_to_image(camera: bpy.types.Object, vertices: Sequence[Vector]) -> List[Tuple[float, float]]:
    scene = bpy.context.scene
    coords = []
    for vertex in vertices:
        co_ndc = bpy_extras.object_utils.world_to_camera_view(scene, camera, vertex)
        x = co_ndc.x * scene.render.resolution_x
        y = (1.0 - co_ndc.y) * scene.render.resolution_y
        coords.append((x, y))
    return coords


def order_corners_clockwise(coords: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not coords:
        return []
    cx = sum(pt[0] for pt in coords) / len(coords)
    cy = sum(pt[1] for pt in coords) / len(coords)
    sorted_coords = sorted(
        coords,
        key=lambda pt: math.atan2(pt[1] - cy, pt[0] - cx),
    )
    start_index = min(range(len(sorted_coords)), key=lambda idx: sorted_coords[idx][0] + sorted_coords[idx][1])
    return sorted_coords[start_index:] + sorted_coords[:start_index]

def create_uniform_principled(name: str, color: Tuple[float, float, float], roughness: float = 0.4) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Specular"].default_value = 0.05
    return mat


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if obj.type != 'MESH':
        return
    if obj.material_slots:
        for slot in obj.material_slots:
            slot.material = material
    else:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def restore_materials(obj: bpy.types.Object, materials: Sequence[Optional[bpy.types.Material]]) -> None:
    if obj.type != 'MESH':
        return
    obj.data.materials.clear()
    for mat in materials:
        if mat is not None:
            obj.data.materials.append(mat)


def capture_mask(scene: bpy.types.Scene, projector_light: bpy.types.Object, file_path: Path) -> float:
    original_filepath = scene.render.filepath
    original_color_mode = scene.render.image_settings.color_mode
    original_color_depth = scene.render.image_settings.color_depth
    original_format = scene.render.image_settings.file_format
    original_exposure = scene.view_settings.exposure

    mask_material = create_uniform_principled("MaskSurface", (1.0, 1.0, 1.0), roughness=0.25)
    saved_materials: dict[str, List[Optional[bpy.types.Material]]] = {}
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        saved_materials[obj.name] = [slot.material for slot in obj.material_slots]
        assign_material(obj, mask_material)

    light_states: List[tuple[str, float, bool]] = []
    for obj in scene.objects:
        if obj.type != 'LIGHT':
            continue
        light_states.append((obj.name, obj.data.energy, obj.hide_render))
        if obj is projector_light:
            obj.hide_render = False
            obj.data.energy *= 1.35
        else:
            obj.hide_render = True

    saved_world = {}
    if scene.world and scene.world.use_nodes:
        bg_node = scene.world.node_tree.nodes.get("Background")
        if bg_node:
            saved_world["color"] = tuple(bg_node.inputs["Color"].default_value)
            saved_world["strength"] = bg_node.inputs["Strength"].default_value
            bg_node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bg_node.inputs["Strength"].default_value = 0.0

    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(file_path)
    scene.view_settings.exposure = 0.0

    bpy.ops.render.render()

    render_result = bpy.data.images.get("Render Result")
    if render_result is None:
        raise RuntimeError("Render Result not available for mask capture")

    pixel_data = list(render_result.pixels[:])
    pixel_count = RESOLUTION_X * RESOLUTION_Y
    luminances = [0.0] * pixel_count
    max_luminance = 0.0
    for idx in range(pixel_count):
        base = idx * 4
        r, g, b = pixel_data[base], pixel_data[base + 1], pixel_data[base + 2]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        luminances[idx] = luminance
        if luminance > max_luminance:
            max_luminance = luminance

    dynamic_threshold = max(0.03, max_luminance * 0.18)
    mask_pixels = [0.0] * (pixel_count * 4)
    for idx in range(pixel_count):
        base = idx * 4
        value = 1.0 if luminances[idx] >= dynamic_threshold else 0.0
        mask_pixels[base] = value
        mask_pixels[base + 1] = value
        mask_pixels[base + 2] = value
        mask_pixels[base + 3] = 1.0

    mask_image = bpy.data.images.new("ProjectionMaskTemp", width=RESOLUTION_X, height=RESOLUTION_Y, alpha=True)
    mask_image.file_format = 'PNG'
    mask_image.pixels.foreach_set(mask_pixels)
    mask_image.filepath_raw = str(file_path)
    mask_image.save()
    bpy.data.images.remove(mask_image)

    for obj_name, materials in saved_materials.items():
        obj = scene.objects.get(obj_name)
        if obj is None:
            continue
        restore_materials(obj, materials)

    for light_name, energy, hide in light_states:
        light_obj = scene.objects.get(light_name)
        if light_obj is None:
            continue
        light_obj.data.energy = energy
        light_obj.hide_render = hide

    if scene.world and scene.world.use_nodes:
        bg_node = scene.world.node_tree.nodes.get("Background")
        if bg_node and "color" in saved_world:
            bg_node.inputs["Color"].default_value = saved_world["color"]
        if bg_node and "strength" in saved_world:
            bg_node.inputs["Strength"].default_value = saved_world["strength"]

    scene.render.image_settings.color_mode = original_color_mode
    scene.render.image_settings.color_depth = original_color_depth
    scene.render.image_settings.file_format = original_format
    scene.render.filepath = original_filepath
    scene.view_settings.exposure = original_exposure

    if mask_material.users == 0:
        bpy.data.materials.remove(mask_material, do_unlink=True)

    return dynamic_threshold


def main() -> None:
    random.seed(42)
    ensure_dirs([OUTPUT_ROOT, COLOR_DIR, MASK_DIR, ANNOTATION_DIR])
    configure_cycles()
    clean_scene()
    scene = bpy.context.scene

    wall_queue: Optional[Deque[Path]] = build_texture_cycle(WALL_TEXTURES) if WALL_TEXTURES else None
    floor_queue: Optional[Deque[Path]] = build_texture_cycle(FLOOR_TEXTURES) if FLOOR_TEXTURES else None

    for index in range(NUM_SAMPLES):
        random.seed(1000 + index)
        clean_scene()
        configure_cycles()
        scene = bpy.context.scene
        scene.frame_current = index

        width = random.uniform(4.0, 6.5)
        depth = random.uniform(4.0, 7.0)
        height = random.uniform(2.5, 3.5)
        wall_texture = next_texture(wall_queue) if wall_queue else None
        floor_texture = next_texture(floor_queue) if floor_queue else None

        room = add_room(width, depth, height, wall_texture)
        add_floor_material(room, floor_texture)

        projection_size = random.uniform(1.2, 2.4)
        projection_ratio = random.uniform(1.4, 1.8)
        desired_width = projection_size * projection_ratio
        desired_height = projection_size
        max_width = max(0.8, width - 0.6)
        max_height = max(0.8, height - 0.6)
        scale_factor = min(1.0, max_width / desired_width, max_height / desired_height)
        projection_width = desired_width * scale_factor
        projection_height = desired_height * scale_factor

        projection_image = random.choice(PROJECTION_IMAGES)
        plane = add_projector_plane(projection_width, projection_height, projection_image)
        plane.location = Vector((
            random.uniform(-width * 0.3, width * 0.3),
            -depth / 2 + 0.02,
            random.uniform(height * 0.35, height * 0.8),
        ))
        plane.rotation_euler = Euler((
            math.radians(90 + random.uniform(-3.0, 3.0)),
            random.uniform(-math.radians(2.0), math.radians(2.0)),
            random.uniform(-math.radians(5.0), math.radians(5.0)),
        ))

        lamp = add_projector_light(plane, depth, projection_image)
        env_info = add_environment_lights((width, depth, height))

        obstacle_target = random.randint(0, 4)
        obstacles = add_obstacles(plane, obstacle_target)

        camera = setup_camera(plane, (width, depth, height))

        bpy.context.view_layer.update()
        polygon_world = [plane.matrix_world @ v.co for v in plane.data.vertices]
        corner_coords = [
            [round(pt[0], 3), round(pt[1], 3)]
            for pt in order_corners_clockwise(project_vertices_to_image(camera, polygon_world))
        ]

        color_path = COLOR_DIR / f"frame_{index:05d}.png"
        mask_path = MASK_DIR / f"frame_{index:05d}.png"
        ann_path = ANNOTATION_DIR / f"frame_{index:05d}.json"

        scene.render.filepath = str(color_path)
        bpy.ops.render.render(write_still=True)

        mask_threshold = capture_mask(scene, lamp, mask_path)

        annotation = {
            "image": color_path.name,
            "mask": mask_path.name,
            "resolution": [RESOLUTION_X, RESOLUTION_Y],
            "seed": 1000 + index,
            "projection_corners": corner_coords,
            "projection_width": projection_width,
            "projection_height": projection_height,
            "room_dimensions": [width, depth, height],
            "wall_texture": wall_texture.name if wall_texture else None,
            "floor_texture": floor_texture.name if floor_texture else None,
            "projection_image": projection_image.name,
            "camera_location": [round(coord, 4) for coord in camera.location],
            "camera_rotation": [round(angle, 5) for angle in camera.rotation_euler],
            "projection_plane_location": [round(coord, 4) for coord in plane.location],
            "projection_plane_rotation": [round(angle, 5) for angle in plane.rotation_euler],
            "projector_location": [round(coord, 4) for coord in lamp.location],
            "projector_energy": round(float(lamp.get("projection_energy", lamp.data.energy)), 2),
            "projector_color": [round(c, 4) for c in lamp.data.color],
            "projector_tint": lamp.get("projection_tint"),
            "projector_spot_degrees": round(math.degrees(lamp.data.spot_size), 3),
            "lighting": env_info,
            "cycles_samples": scene.cycles.samples,
            "color_management": {
                "exposure": round(scene.view_settings.exposure, 4),
                "gamma": round(scene.view_settings.gamma, 4),
            },
            "mask_threshold": round(mask_threshold, 6),
            "materials": {
                "wall": "texture" if wall_texture else "procedural",
                "floor": "texture" if floor_texture else "procedural",
            },
            "obstacle_count": len(obstacles),
            "obstacles": [
                {
                    "shape": obj.get("shape", obj.name),
                    "location": [round(coord, 4) for coord in obj.location],
                    "scale": [round(scale, 4) for scale in obj.scale],
                }
                for obj in obstacles
            ],
        }
        ann_path.write_text(json.dumps(annotation, indent=2))

        if (index + 1) % 100 == 0:
            print(f"Progreso: {index + 1}/{NUM_SAMPLES}")

    print(f"Dataset generated at {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
