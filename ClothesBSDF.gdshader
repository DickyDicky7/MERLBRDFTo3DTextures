    shader_type spatial;
//  shader_type spatial;
    render_mode blend_mix, depth_draw_opaque, cull_disabled, specular_disabled;
//  render_mode blend_mix, depth_draw_opaque, cull_disabled, specular_disabled;

    // --- 3D BSDF Look-Up Table ---
//  // --- 3D BSDF Look-Up Table ---
    // The 3D texture containing the baked PBRT Disney Cloth Bidirectional Scattering Distribution Function (BSDF).
//  // The 3D texture containing the baked PBRT Disney Cloth Bidirectional Scattering Distribution Function (BSDF).
    // This Look-Up Table (LUT) stores the precomputed complex cloth scattering properties in the following channel layout:
//  // This Look-Up Table (LUT) stores the precomputed complex cloth scattering properties in the following channel layout:
    // R: Base Lobes (sum of Disney Diffuse, Retro-reflection, and Fake Subsurface scattering)
//  // R: Base Lobes (sum of Disney Diffuse, Retro-reflection, and Fake Subsurface scattering)
    // G: Sheen Lobe (Disney Sheen for edge highlights)
//  // G: Sheen Lobe (Disney Sheen for edge highlights)
    // B: Unused (reserved, always 0.0)
//  // B: Unused (reserved, always 0.0)
    // It must be imported into Godot as a 3D texture with dimensions of 64x64x64 (8 columns, 8 rows).
//  // It must be imported into Godot as a 3D texture with dimensions of 64x64x64 (8 columns, 8 rows).
    uniform sampler3D cloth_bsdf_texture_lut : hint_default_black, filter_linear, repeat_disable;
//  uniform sampler3D cloth_bsdf_texture_lut : hint_default_black, filter_linear, repeat_disable;

    // --- Basic Cloth Material Properties ---
//  // --- Basic Cloth Material Properties ---
    // Defines the foundational visual characteristics of the cloth surface.
//  // Defines the foundational visual characteristics of the cloth surface.
    group_uniforms base_color;
//  group_uniforms base_color;
    uniform vec3 cloth_base_albedo_color : source_color = vec3(1.0, 1.0, 1.0);
//  uniform vec3 cloth_base_albedo_color : source_color = vec3(1.0, 1.0, 1.0);
    uniform vec3 cloth_sheen_tint_color : source_color = vec3(1.0, 1.0, 1.0);
//  uniform vec3 cloth_sheen_tint_color : source_color = vec3(1.0, 1.0, 1.0);
    // Controls the overall intensity of the sheen scattering lobe, providing artist control over the edge highlight strength.
//  // Controls the overall intensity of the sheen scattering lobe, providing artist control over the edge highlight strength.
    uniform float cloth_sheen_intensity : hint_range(0.0, 5.0) = 1.0;
//  uniform float cloth_sheen_intensity : hint_range(0.0, 5.0) = 1.0;
    uniform sampler2D cloth_detail_albedo_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D cloth_detail_albedo_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
    uniform vec2 cloth_detail_texture_uv_coordinate_scale = vec2(1.0, 1.0);
//  uniform vec2 cloth_detail_texture_uv_coordinate_scale = vec2(1.0, 1.0);
    group_uniforms;
//  group_uniforms;

    // --- Micro-Surface Normal Properties ---
//  // --- Micro-Surface Normal Properties ---
    // Controls the fine surface details and normal-mapped interactions with light.
//  // Controls the fine surface details and normal-mapped interactions with light.
    group_uniforms normal_map;
//  group_uniforms normal_map;
    uniform sampler2D cloth_normal_map_texture : hint_normal, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D cloth_normal_map_texture : hint_normal, filter_linear_mipmap, repeat_enable;
    uniform float cloth_normal_map_intensity : hint_range(0.0, 5.0) = 1.0;
//  uniform float cloth_normal_map_intensity : hint_range(0.0, 5.0) = 1.0;
    uniform vec2 cloth_normal_map_texture_uv_coordinate_scale = vec2(1.0, 1.0);
//  uniform vec2 cloth_normal_map_texture_uv_coordinate_scale = vec2(1.0, 1.0);
    group_uniforms;
//  group_uniforms;

    // --- Advanced Lighting and Shading Properties ---
//  // --- Advanced Lighting and Shading Properties ---
    // Controls how the cloth reacts to both direct light sources and indirect ambient lighting.
//  // Controls how the cloth reacts to both direct light sources and indirect ambient lighting.
    group_uniforms lighting;
//  group_uniforms lighting;
    uniform float cloth_direct_light_intensity_scale : hint_range(0.0, 10.0) = 1.0;
//  uniform float cloth_direct_light_intensity_scale : hint_range(0.0, 10.0) = 1.0;
    uniform float cloth_indirect_ambient_albedo_scale : hint_range(0.0, 1.0) = 1.0;
//  uniform float cloth_indirect_ambient_albedo_scale : hint_range(0.0, 1.0) = 1.0;
    group_uniforms;
//  group_uniforms;

    // --- Ambient Occlusion Properties ---
//  // --- Ambient Occlusion Properties ---
    // Adds localized self-shadowing to darken creases, folds, and recessed areas of the cloth surface.
//  // Adds localized self-shadowing to darken creases, folds, and recessed areas of the cloth surface.
    group_uniforms ambient_occlusion;
//  group_uniforms ambient_occlusion;
    uniform bool enable_cloth_ambient_occlusion = false;
//  uniform bool enable_cloth_ambient_occlusion = false;
    uniform sampler2D cloth_ambient_occlusion_texture : hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D cloth_ambient_occlusion_texture : hint_default_white, filter_linear_mipmap, repeat_enable;
    uniform float cloth_ambient_occlusion_intensity : hint_range(0.0, 1.0) = 1.0;
//  uniform float cloth_ambient_occlusion_intensity : hint_range(0.0, 1.0) = 1.0;
    // The scale factor used to tile the ambient occlusion texture across the mesh surface.
//  // The scale factor used to tile the ambient occlusion texture across the mesh surface.
    uniform vec2 cloth_ambient_occlusion_uv_coordinate_scale = vec2(1.0, 1.0);
//  uniform vec2 cloth_ambient_occlusion_uv_coordinate_scale = vec2(1.0, 1.0);
    group_uniforms;
//  group_uniforms;

    // --- Backface Light Transmission Properties ---
//  // --- Backface Light Transmission Properties ---
    // Controls the transmission of light through thin cloth geometry, such as curtains, flags, or loose draped fabric.
//  // Controls the transmission of light through thin cloth geometry, such as curtains, flags, or loose draped fabric.
    group_uniforms transmission;
//  group_uniforms transmission;
    uniform bool enable_cloth_backface_transmission = false;
//  uniform bool enable_cloth_backface_transmission = false;
    // Controls the intensity of light passing through the cloth volume from behind.
//  // Controls the intensity of light passing through the cloth volume from behind.
    uniform float cloth_backface_transmission_intensity : hint_range(0.0, 2.0) = 0.5;
//  uniform float cloth_backface_transmission_intensity : hint_range(0.0, 2.0) = 0.5;
    // The color tint applied to transmitted light, simulating absorption by the cloth fibers.
//  // The color tint applied to transmitted light, simulating absorption by the cloth fibers.
    uniform vec3 cloth_backface_transmission_tint_color : source_color = vec3(1.0, 1.0, 1.0);
//  uniform vec3 cloth_backface_transmission_tint_color : source_color = vec3(1.0, 1.0, 1.0);
    group_uniforms;
//  group_uniforms;

    // Precomputed constants for the half-texel correction specific to the 64x64x64 LUT dimensions.
//  // Precomputed constants for the half-texel correction specific to the 64x64x64 LUT dimensions.
    // These constants prevent sampling artifacts and interpolation errors near the edges of the 3D texture volume.
//  // These constants prevent sampling artifacts and interpolation errors near the edges of the 3D texture volume.
    const float constant_lut_half_texel_scale = 0.984375;
//  const float constant_lut_half_texel_scale = 0.984375;
    const float constant_lut_half_texel_offset = 0.0078125;
//  const float constant_lut_half_texel_offset = 0.0078125;

    // Varying variables used to pass calculated data from the fragment function to the per-light function.
//  // Varying variables used to pass calculated data from the fragment function to the per-light function.
    varying vec3 stored_cloth_raw_base_color;
//  varying vec3 stored_cloth_raw_base_color;

    // Pre-scaled UV coordinate varyings computed once per vertex to avoid redundant per-fragment multiplication.
//  // Pre-scaled UV coordinate varyings computed once per vertex to avoid redundant per-fragment multiplication.
    varying vec2 scaled_cloth_detail_albedo_uv;
//  varying vec2 scaled_cloth_detail_albedo_uv;
    varying vec2 scaled_cloth_normal_map_uv;
//  varying vec2 scaled_cloth_normal_map_uv;
    varying vec2 scaled_cloth_ambient_occlusion_uv;
//  varying vec2 scaled_cloth_ambient_occlusion_uv;

    void vertex() {
//  void vertex() {
        // Pre-scales UV coordinates in the vertex stage to avoid redundant per-fragment multiplication.
//      // Pre-scales UV coordinates in the vertex stage to avoid redundant per-fragment multiplication.
        scaled_cloth_detail_albedo_uv = UV * cloth_detail_texture_uv_coordinate_scale;
//      scaled_cloth_detail_albedo_uv = UV * cloth_detail_texture_uv_coordinate_scale;
        scaled_cloth_normal_map_uv = UV * cloth_normal_map_texture_uv_coordinate_scale;
//      scaled_cloth_normal_map_uv = UV * cloth_normal_map_texture_uv_coordinate_scale;
        scaled_cloth_ambient_occlusion_uv = UV * cloth_ambient_occlusion_uv_coordinate_scale;
//      scaled_cloth_ambient_occlusion_uv = UV * cloth_ambient_occlusion_uv_coordinate_scale;
    }
//  }

    void fragment() {
//  void fragment() {
        // Samples the full detail albedo texel using the pre-scaled UV varying variable.
//      // Samples the full detail albedo texel using the pre-scaled UV varying variable.
        vec4 sampled_detail_albedo_texel = texture(cloth_detail_albedo_texture, scaled_cloth_detail_albedo_uv);
//      vec4 sampled_detail_albedo_texel = texture(cloth_detail_albedo_texture, scaled_cloth_detail_albedo_uv);

        // Samples the normal map texture using the pre-scaled UV varying variable.
//      // Samples the normal map texture using the pre-scaled UV varying variable.
        vec3 sampled_cloth_normal_map_color = texture(cloth_normal_map_texture, scaled_cloth_normal_map_uv).rgb;
//      vec3 sampled_cloth_normal_map_color = texture(cloth_normal_map_texture, scaled_cloth_normal_map_uv).rgb;

        // Applies the sampled normal map to the built-in PBR normal map property to alter the surface normal for lighting calculations.
//      // Applies the sampled normal map to the built-in PBR normal map property to alter the surface normal for lighting calculations.
        NORMAL_MAP = sampled_cloth_normal_map_color;
//      NORMAL_MAP = sampled_cloth_normal_map_color;

        // Scales the overall intensity of the normal map effect.
//      // Scales the overall intensity of the normal map effect.
        NORMAL_MAP_DEPTH = cloth_normal_map_intensity;
//      NORMAL_MAP_DEPTH = cloth_normal_map_intensity;

        // Multiplies the uniform base color by the sampled detail texture to produce the final cloth albedo base color.
//      // Multiplies the uniform base color by the sampled detail texture to produce the final cloth albedo base color.
        vec3 calculated_base_color = cloth_base_albedo_color * sampled_detail_albedo_texel.rgb;
//      vec3 calculated_base_color = cloth_base_albedo_color * sampled_detail_albedo_texel.rgb;

        // --- Ambient Occlusion Calculation ---
//      // --- Ambient Occlusion Calculation ---
        // When enabled, applies ambient occlusion to add localized shadowing in creases and folds.
//      // When enabled, applies ambient occlusion to add localized shadowing in creases and folds.
        if (enable_cloth_ambient_occlusion) {
//      if (enable_cloth_ambient_occlusion) {
            // Samples the ambient occlusion texture using the pre-scaled UV varying variable.
//          // Samples the ambient occlusion texture using the pre-scaled UV varying variable.
            float sampled_cloth_ambient_occlusion = texture(cloth_ambient_occlusion_texture, scaled_cloth_ambient_occlusion_uv).r;
//          float sampled_cloth_ambient_occlusion = texture(cloth_ambient_occlusion_texture, scaled_cloth_ambient_occlusion_uv).r;

            // Interpolates between the unoccluded baseline (1.0) and the sampled value based on the intensity uniform.
//          // Interpolates between the unoccluded baseline (1.0) and the sampled value based on the intensity uniform.
            AO = mix(1.0, sampled_cloth_ambient_occlusion, cloth_ambient_occlusion_intensity);
//          AO = mix(1.0, sampled_cloth_ambient_occlusion, cloth_ambient_occlusion_intensity);

            // Ensures the ambient occlusion also affects direct light sources, deepening the shadowing of folds and recessed geometry.
//          // Ensures the ambient occlusion also affects direct light sources, deepening the shadowing of folds and recessed geometry.
            AO_LIGHT_AFFECT = 1.0;
//          AO_LIGHT_AFFECT = 1.0;
        }
//      }

        // Stores the raw base color as a varying variable for use in the per-light function.
//      // Stores the raw base color as a varying variable for use in the per-light function.
        // By passing this as a varying, the shader avoids redundant texture sampling in the per-light function.
//      // By passing this as a varying, the shader avoids redundant texture sampling in the per-light function.
        stored_cloth_raw_base_color = calculated_base_color;
//      stored_cloth_raw_base_color = calculated_base_color;

        // The albedo drives the base color in Godot's forward pass for ambient and indirect light calculations.
//      // The albedo drives the base color in Godot's forward pass for ambient and indirect light calculations.
        // It is scaled by the indirect ambient albedo scale uniform to ensure ambient light does not unnaturally wash out the cloth texture.
//      // It is scaled by the indirect ambient albedo scale uniform to ensure ambient light does not unnaturally wash out the cloth texture.
        ALBEDO = calculated_base_color * cloth_indirect_ambient_albedo_scale;
//      ALBEDO = calculated_base_color * cloth_indirect_ambient_albedo_scale;

        // Overrides the default PBR metallic and roughness values.
//      // Overrides the default PBR metallic and roughness values.
        // Setting ROUGHNESS to 1.0 and METALLIC to 0.0 prevents Godot's standard reflections from interfering with the custom BSDF lighting.
//      // Setting ROUGHNESS to 1.0 and METALLIC to 0.0 prevents Godot's standard reflections from interfering with the custom BSDF lighting.
        ROUGHNESS = 1.0;
//      ROUGHNESS = 1.0;
        METALLIC = 0.0;
//      METALLIC = 0.0;
    }
//  }

    void light() {
//  void light() {
        // Retrieves the pre-normalized vectors from Godot's built-in light function inputs.
//      // Retrieves the pre-normalized vectors from Godot's built-in light function inputs.
        vec3 light_direction = LIGHT;
//      vec3 light_direction = LIGHT;
        vec3 view_direction = VIEW;
//      vec3 view_direction = VIEW;
        vec3 surface_normal = NORMAL;
//      vec3 surface_normal = NORMAL;

        // Calculates the cosine of the angle between the surface normal and the incident light direction.
//      // Calculates the cosine of the angle between the surface normal and the incident light direction.
        // This determines how directly the light strikes the cloth surface (clamped to 0.0 to exclude back-face illumination).
//      // This determines how directly the light strikes the cloth surface (clamped to 0.0 to exclude back-face illumination).
        float dot_product_normal_and_light = max(0.0, dot(surface_normal, light_direction));
//      float dot_product_normal_and_light = max(0.0, dot(surface_normal, light_direction));

        // Calculates the cosine of the angle between the surface normal and the outgoing view direction.
//      // Calculates the cosine of the angle between the surface normal and the outgoing view direction.
        // This captures grazing-angle effects such as sheen and Fresnel falloff (clamped to 0.0 to exclude back-face illumination).
//      // This captures grazing-angle effects such as sheen and Fresnel falloff (clamped to 0.0 to exclude back-face illumination).
        float dot_product_normal_and_view = max(0.0, dot(surface_normal, view_direction));
//      float dot_product_normal_and_view = max(0.0, dot(surface_normal, view_direction));

        // Calculates the dot product between the light and view directions.
//      // Calculates the dot product between the light and view directions.
        // This captures the angular relationship for retro-reflection and forward scattering (full range -1.0 to 1.0).
//      // This captures the angular relationship for retro-reflection and forward scattering (full range -1.0 to 1.0).
        // Clamps the dot product to the [-1.0, 1.0] range to guard against floating-point rounding overshoot.
//      // Clamps the dot product to the [-1.0, 1.0] range to guard against floating-point rounding overshoot.
        float dot_product_light_and_view = clamp(dot(light_direction, view_direction), -1.0, 1.0);
//      float dot_product_light_and_view = clamp(dot(light_direction, view_direction), -1.0, 1.0);

        // --- Map Dot Products to Normalized 3D Texture Coordinates ---
//      // --- Map Dot Products to Normalized 3D Texture Coordinates ---
        // Maps the calculated dot products to the normalized texture coordinates [0.0, 1.0] for the U, V, and W axes of the 3D LUT.
//      // Maps the calculated dot products to the normalized texture coordinates [0.0, 1.0] for the U, V, and W axes of the 3D LUT.

        // Maps dot_product_normal_and_light (range 0..1) directly to the U texture coordinate.
//      // Maps dot_product_normal_and_light (range 0..1) directly to the U texture coordinate.
        float texture_coordinate_u = dot_product_normal_and_light;
//      float texture_coordinate_u = dot_product_normal_and_light;

        // Maps dot_product_normal_and_view (range 0..1) directly to the V texture coordinate.
//      // Maps dot_product_normal_and_view (range 0..1) directly to the V texture coordinate.
        float texture_coordinate_v = dot_product_normal_and_view;
//      float texture_coordinate_v = dot_product_normal_and_view;

        // Remaps dot_product_light_and_view from the range [-1, 1] to the normalized [0, 1] range for the W texture coordinate.
//      // Remaps dot_product_light_and_view from the range [-1, 1] to the normalized [0, 1] range for the W texture coordinate.
        float texture_coordinate_w = dot_product_light_and_view * 0.5 + 0.5;
//      float texture_coordinate_w = dot_product_light_and_view * 0.5 + 0.5;

        // Applies a half-texel correction using precomputed constants specific to the 64x64x64 LUT dimensions.
//      // Applies a half-texel correction using precomputed constants specific to the 64x64x64 LUT dimensions.
        // This prevents sampling artifacts and interpolation errors near the edges of the 3D texture volume.
//      // This prevents sampling artifacts and interpolation errors near the edges of the 3D texture volume.
        vec3 texture_coordinates_3d = vec3(texture_coordinate_u, texture_coordinate_v, texture_coordinate_w);
//      vec3 texture_coordinates_3d = vec3(texture_coordinate_u, texture_coordinate_v, texture_coordinate_w);
        texture_coordinates_3d = (texture_coordinates_3d * constant_lut_half_texel_scale) + constant_lut_half_texel_offset;
//      texture_coordinates_3d = (texture_coordinates_3d * constant_lut_half_texel_scale) + constant_lut_half_texel_offset;

        // --- Sample the Pre-baked BSDF Look-Up Table ---
//      // --- Sample the Pre-baked BSDF Look-Up Table ---
        // Evaluates the cloth scattering function by sampling the 3D texture at the calculated (U, V, W) coordinates.
//      // Evaluates the cloth scattering function by sampling the 3D texture at the calculated (U, V, W) coordinates.
        // R: Base Lobes (Diffuse, Retro, Fake Subsurface)
//      // R: Base Lobes (Diffuse, Retro, Fake Subsurface)
        // G: Sheen Lobe
//      // G: Sheen Lobe
        // B: Unused
//      // B: Unused
        vec3 sampled_brdf_evaluation = texture(cloth_bsdf_texture_lut, texture_coordinates_3d).rgb;
//      vec3 sampled_brdf_evaluation = texture(cloth_bsdf_texture_lut, texture_coordinates_3d).rgb;

        // --- Combine the Evaluated Light Lobes ---
//      // --- Combine the Evaluated Light Lobes ---
        // Scales the raw base color by the direct light intensity scale uniform to provide artist control over the direct lighting response.
//      // Scales the raw base color by the direct light intensity scale uniform to provide artist control over the direct lighting response.
        vec3 scaled_cloth_direct_base_color = stored_cloth_raw_base_color * cloth_direct_light_intensity_scale;
//      vec3 scaled_cloth_direct_base_color = stored_cloth_raw_base_color * cloth_direct_light_intensity_scale;

        // Multiplies the base scattering lobes (diffuse, retro-reflection, and fake subsurface) by the scaled base color.
//      // Multiplies the base scattering lobes (diffuse, retro-reflection, and fake subsurface) by the scaled base color.
        vec3 base_scattering = sampled_brdf_evaluation.r * scaled_cloth_direct_base_color;
//      vec3 base_scattering = sampled_brdf_evaluation.r * scaled_cloth_direct_base_color;

        // Multiplies the sheen scattering lobe by the sheen tint color, sheen intensity, and direct light intensity scale to provide consistent artist control over the sheen response.
//      // Multiplies the sheen scattering lobe by the sheen tint color, sheen intensity, and direct light intensity scale to provide consistent artist control over the sheen response.
        vec3 sheen_scattering = sampled_brdf_evaluation.g * cloth_sheen_tint_color * cloth_sheen_intensity * cloth_direct_light_intensity_scale;
//      vec3 sheen_scattering = sampled_brdf_evaluation.g * cloth_sheen_tint_color * cloth_sheen_intensity * cloth_direct_light_intensity_scale;

        // The LUT stores the exact BRDF evaluation. The result is multiplied by the incoming light intensity, attenuation, and N dot L to complete the rendering equation.
//      // The LUT stores the exact BRDF evaluation. The result is multiplied by the incoming light intensity, attenuation, and N dot L to complete the rendering equation.
        vec3 combined_final_light = (base_scattering + sheen_scattering) * LIGHT_COLOR * ATTENUATION * dot_product_normal_and_light;
//      vec3 combined_final_light = (base_scattering + sheen_scattering) * LIGHT_COLOR * ATTENUATION * dot_product_normal_and_light;

        DIFFUSE_LIGHT += combined_final_light;
//      DIFFUSE_LIGHT += combined_final_light;

        // --- Backface Light Transmission ---
//      // --- Backface Light Transmission ---
        // Simulates light passing through thin cloth geometry using a wrap lighting model.
//      // Simulates light passing through thin cloth geometry using a wrap lighting model.
        if (enable_cloth_backface_transmission) {
//      if (enable_cloth_backface_transmission) {
            // Computes the raw (unclamped) N dot L dot product to capture back-face illumination that the clamped version discards.
//          // Computes the raw (unclamped) N dot L dot product to capture back-face illumination that the clamped version discards.
            float raw_dot_product_normal_and_light = dot(surface_normal, light_direction);
//          float raw_dot_product_normal_and_light = dot(surface_normal, light_direction);

            // Applies a wrap lighting model to soften the terminator and allow light to wrap around the geometry.
//          // Applies a wrap lighting model to soften the terminator and allow light to wrap around the geometry.
            const float constant_cloth_transmission_wrap_factor = 0.5;
//          const float constant_cloth_transmission_wrap_factor = 0.5;
            float calculated_cloth_wrap_transmission_dot = max(-raw_dot_product_normal_and_light + constant_cloth_transmission_wrap_factor, 0.0) / (1.0 + constant_cloth_transmission_wrap_factor);
//          float calculated_cloth_wrap_transmission_dot = max(-raw_dot_product_normal_and_light + constant_cloth_transmission_wrap_factor, 0.0) / (1.0 + constant_cloth_transmission_wrap_factor);

            // Squares the wrap factor for a softer falloff and modulates it by view-light alignment for physically plausible forward scattering.
//          // Squares the wrap factor for a softer falloff and modulates it by view-light alignment for physically plausible forward scattering.
            float computed_cloth_backlight_scatter = (calculated_cloth_wrap_transmission_dot * calculated_cloth_wrap_transmission_dot) * max(-dot_product_light_and_view, 0.0);
//          float computed_cloth_backlight_scatter = (calculated_cloth_wrap_transmission_dot * calculated_cloth_wrap_transmission_dot) * max(-dot_product_light_and_view, 0.0);

            // Combines the transmission scatter with the base color, tint color, and intensity to produce the final transmitted light contribution.
//          // Combines the transmission scatter with the base color, tint color, and intensity to produce the final transmitted light contribution.
            vec3 evaluated_cloth_transmission = stored_cloth_raw_base_color * cloth_backface_transmission_tint_color * cloth_backface_transmission_intensity * LIGHT_COLOR * computed_cloth_backlight_scatter;
//          vec3 evaluated_cloth_transmission = stored_cloth_raw_base_color * cloth_backface_transmission_tint_color * cloth_backface_transmission_intensity * LIGHT_COLOR * computed_cloth_backlight_scatter;

            // Blends in a small baseline amount of transmission even in shadowed areas to simulate global multiple scattering through the cloth.
//          // Blends in a small baseline amount of transmission even in shadowed areas to simulate global multiple scattering through the cloth.
            DIFFUSE_LIGHT += evaluated_cloth_transmission * mix(0.1, 1.0, ATTENUATION);
//          DIFFUSE_LIGHT += evaluated_cloth_transmission * mix(0.1, 1.0, ATTENUATION);
        }
//      }
    }
//  }
