    shader_type spatial;
//  shader_type spatial;
    render_mode blend_mix, depth_draw_opaque, cull_disabled, diffuse_burley, specular_schlick_ggx;
//  render_mode blend_mix, depth_draw_opaque, cull_disabled, diffuse_burley, specular_schlick_ggx;

    // The 3D texture containing the baked PBRT Disney Cloth Bidirectional Scattering Distribution Function (BSDF)
//  // The 3D texture containing the baked PBRT Disney Cloth Bidirectional Scattering Distribution Function (BSDF)
    // This Look-Up Table (LUT) stores the precomputed complex cloth scattering properties (Diffuse, Retro, Fake Subsurface, Sheen).
//  // This Look-Up Table (LUT) stores the precomputed complex cloth scattering properties (Diffuse, Retro, Fake Subsurface, Sheen).
    // This should be imported in Godot as a 3D Texture with dimensions 64x64x64 (8 columns, 8 rows).
//  // This should be imported in Godot as a 3D Texture with dimensions 64x64x64 (8 columns, 8 rows).
    uniform sampler3D clothes_bsdf_texture_lut : hint_default_black, filter_linear, repeat_disable;
//  uniform sampler3D clothes_bsdf_texture_lut : hint_default_black, filter_linear, repeat_disable;

    // Basic Cloth Material Properties
//  // Basic Cloth Material Properties
    // Defines the foundational visual characteristics of the cloth surface.
//  // Defines the foundational visual characteristics of the cloth surface.
    uniform vec3 base_albedo_color : source_color = vec3(1.0, 1.0, 1.0);
//  uniform vec3 base_albedo_color : source_color = vec3(1.0, 1.0, 1.0);
    uniform vec3 sheen_tint_color : source_color = vec3(1.0, 1.0, 1.0);
//  uniform vec3 sheen_tint_color : source_color = vec3(1.0, 1.0, 1.0);
    uniform sampler2D detail_albedo_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D detail_albedo_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
    uniform vec2 detail_texture_uv_scale = vec2(1.0, 1.0);
//  uniform vec2 detail_texture_uv_scale = vec2(1.0, 1.0);

    // Micro-Surface Normal Properties
//  // Micro-Surface Normal Properties
    // Controls the fine surface details and bumping interactions with light.
//  // Controls the fine surface details and bumping interactions with light.
    uniform sampler2D detail_normal_texture : hint_normal, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D detail_normal_texture : hint_normal, filter_linear_mipmap, repeat_enable;
    uniform float detail_normal_depth : hint_range(0.0, 5.0) = 1.0;
//  uniform float detail_normal_depth : hint_range(0.0, 5.0) = 1.0;
    uniform vec2 detail_normal_uv_scale = vec2(1.0, 1.0);
//  uniform vec2 detail_normal_uv_scale = vec2(1.0, 1.0);

    // Advanced Lighting and Shading Properties
//  // Advanced Lighting and Shading Properties
    // Controls how the cloth reacts to both direct light sources and indirect ambient lighting.
//  // Controls how the cloth reacts to both direct light sources and indirect ambient lighting.
    uniform float direct_light_albedo_scale : hint_range(1.0, 100.0) = 1.0;
//  uniform float direct_light_albedo_scale : hint_range(1.0, 100.0) = 1.0;
    uniform float indirect_ambient_albedo_scale : hint_range(0.0, 1.0) = 1.0;
//  uniform float indirect_ambient_albedo_scale : hint_range(0.0, 1.0) = 1.0;

    // Ambient Occlusion Properties
//  // Ambient Occlusion Properties
    // Adds localized shadowing in creases and folds to ground the cloth visually.
//  // Adds localized shadowing in creases and folds to ground the cloth visually.
    uniform bool enable_ambient_occlusion = false;
//  uniform bool enable_ambient_occlusion = false;
    uniform sampler2D ambient_occlusion_texture : hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D ambient_occlusion_texture : hint_default_white, filter_linear_mipmap, repeat_enable;
    uniform float ambient_occlusion_strength : hint_range(0.0, 1.0) = 1.0;
//  uniform float ambient_occlusion_strength : hint_range(0.0, 1.0) = 1.0;

    varying vec3 stored_raw_base_color;
//  varying vec3 stored_raw_base_color;

    void fragment() {
//  void fragment() {
        vec4 sampled_detail_texture = texture(detail_albedo_texture, UV * detail_texture_uv_scale);
//      vec4 sampled_detail_texture = texture(detail_albedo_texture, UV * detail_texture_uv_scale);

        // Sample and apply the normal map to alter the surface normal for lighting calculation.
//      // Sample and apply the normal map to alter the surface normal for lighting calculation.
        vec3 sampled_normal_texture = texture(detail_normal_texture, UV * detail_normal_uv_scale).rgb;
//      vec3 sampled_normal_texture = texture(detail_normal_texture, UV * detail_normal_uv_scale).rgb;
        NORMAL_MAP = sampled_normal_texture;
//      NORMAL_MAP = sampled_normal_texture;
        NORMAL_MAP_DEPTH = detail_normal_depth;
//      NORMAL_MAP_DEPTH = detail_normal_depth;
        vec3 calculated_base_color = base_albedo_color * sampled_detail_texture.rgb;
//      vec3 calculated_base_color = base_albedo_color * sampled_detail_texture.rgb;

        // Apply Ambient Occlusion if enabled.
//      // Apply Ambient Occlusion if enabled.
        if (enable_ambient_occlusion) {
//      if (enable_ambient_occlusion) {
            float sampled_ao = texture(ambient_occlusion_texture, UV * detail_texture_uv_scale).r;
//          float sampled_ao = texture(ambient_occlusion_texture, UV * detail_texture_uv_scale).r;
            AO = mix(1.0, sampled_ao, ambient_occlusion_strength);
//          AO = mix(1.0, sampled_ao, ambient_occlusion_strength);
            AO_LIGHT_AFFECT = 1.0;
//          AO_LIGHT_AFFECT = 1.0;
        }
//      }

        // Store the raw base color for the light() function to optimize performance.
//      // Store the raw base color for the light() function to optimize performance.
        // By passing this as a varying, we avoid redundant texture sampling in the per-pixel lighting pass.
//      // By passing this as a varying, we avoid redundant texture sampling in the per-pixel lighting pass.
        stored_raw_base_color = calculated_base_color;
//      stored_raw_base_color = calculated_base_color;

        // Albedo drives the base color in Godot's forward pass for ambient and indirect light calculation.
//      // Albedo drives the base color in Godot's forward pass for ambient and indirect light calculation.
        // We scale it by the indirect_ambient_albedo_scale uniform to ensure ambient light does not unnaturally wash out the cloth texture.
//      // We scale it by the indirect_ambient_albedo_scale uniform to ensure ambient light does not unnaturally wash out the cloth texture.
        ALBEDO = calculated_base_color * indirect_ambient_albedo_scale;
//      ALBEDO = calculated_base_color * indirect_ambient_albedo_scale;

        // Ensure we do not output default PBR metallic or roughness values.
//      // Ensure we do not output default PBR metallic or roughness values.
        // Setting ROUGHNESS to 1.0 and METALLIC to 0.0 prevents Godot's standard reflections from interfering with our custom BSDF lighting function.
//      // Setting ROUGHNESS to 1.0 and METALLIC to 0.0 prevents Godot's standard reflections from interfering with our custom BSDF lighting function.
        ROUGHNESS = 1.0;
//      ROUGHNESS = 1.0;
        METALLIC = 0.0;
//      METALLIC = 0.0;
    }
//  }

    void light() {
//  void light() {
        vec3 light_direction = normalize(LIGHT);
//      vec3 light_direction = normalize(LIGHT);
        vec3 view_direction = normalize(VIEW);
//      vec3 view_direction = normalize(VIEW);
        vec3 surface_normal = normalize(NORMAL);
//      vec3 surface_normal = normalize(NORMAL);

        float ndotl = max(0.0, dot(surface_normal, light_direction));
//      float ndotl = max(0.0, dot(surface_normal, light_direction));
        float ndotv = max(0.0, dot(surface_normal, view_direction));
//      float ndotv = max(0.0, dot(surface_normal, view_direction));
        float ldotv = clamp(dot(light_direction, view_direction), -1.0, 1.0);
//      float ldotv = clamp(dot(light_direction, view_direction), -1.0, 1.0);

        // Map the dot products to the normalized texture coordinates [0.0, 1.0] for U, V, and W axes.
//      // Map the dot products to the normalized texture coordinates [0.0, 1.0] for U, V, and W axes.
        // Map ndotl (0..1) -> U, ndotv (0..1) -> V, ldotv (-1..1) -> W
//      // Map ndotl (0..1) -> U, ndotv (0..1) -> V, ldotv (-1..1) -> W
        float u = ndotl;
//      float u = ndotl;
        float v = ndotv;
//      float v = ndotv;
        float w = ldotv * 0.5 + 0.5;
//      float w = ldotv * 0.5 + 0.5;

        // Apply a half-texel correction specific to our 64x64x64 LUT dimensions.
//      // Apply a half-texel correction specific to our 64x64x64 LUT dimensions.
        // This prevents sampling artifacts and interpolation errors near the edges of the 3D texture volume.
//      // This prevents sampling artifacts and interpolation errors near the edges of the 3D texture volume.
        float correction = 1.0 / 64.0;
//      float correction = 1.0 / 64.0;
        vec3 tex_coords = vec3(u, v, w) * (1.0 - correction) + (correction * 0.5);
//      vec3 tex_coords = vec3(u, v, w) * (1.0 - correction) + (correction * 0.5);

        // Sample the Pre-baked BSDF Look-Up Table
//      // Sample the Pre-baked BSDF Look-Up Table
        // We evaluate the complex cloth scattering function by sampling the 3D texture at the calculated (U, V, W) coordinates.
//      // We evaluate the complex cloth scattering function by sampling the 3D texture at the calculated (U, V, W) coordinates.
        // R: Base Lobes (Diffuse, Retro, Fake Subsurface)
//      // R: Base Lobes (Diffuse, Retro, Fake Subsurface)
        // G: Sheen Lobe
//      // G: Sheen Lobe
        // B: Unused
//      // B: Unused
        vec3 sampled_brdf = texture(clothes_bsdf_texture_lut, tex_coords).rgb;
//      vec3 sampled_brdf = texture(clothes_bsdf_texture_lut, tex_coords).rgb;

        // Combine the evaluated light lobes.
//      // Combine the evaluated light lobes.
        vec3 scaled_base_color = stored_raw_base_color * direct_light_albedo_scale;
//      vec3 scaled_base_color = stored_raw_base_color * direct_light_albedo_scale;

        // Base scattering multiplied by the base color.
//      // Base scattering multiplied by the base color.
        vec3 base_scattering = sampled_brdf.r * scaled_base_color;
//      vec3 base_scattering = sampled_brdf.r * scaled_base_color;

        // Sheen scattering multiplied by the sheen tint color.
//      // Sheen scattering multiplied by the sheen tint color.
        vec3 sheen_scattering = sampled_brdf.g * sheen_tint_color;
//      vec3 sheen_scattering = sampled_brdf.g * sheen_tint_color;

        // The LUT stores the exact BRDF evaluation. We must multiply by the incoming light intensity, attenuation, and ndotl to complete the rendering equation.
//      // The LUT stores the exact BRDF evaluation. We must multiply by the incoming light intensity, attenuation, and ndotl to complete the rendering equation.
        vec3 combined_light = (base_scattering + sheen_scattering) * LIGHT_COLOR * ATTENUATION * ndotl;
//      vec3 combined_light = (base_scattering + sheen_scattering) * LIGHT_COLOR * ATTENUATION * ndotl;

        DIFFUSE_LIGHT += combined_light;
//      DIFFUSE_LIGHT += combined_light;
    }
//  }
