    shader_type spatial;
//  shader_type spatial;
    render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;
//  render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;

    // The 2D Pre-Integrated Skin Look-Up Table (LUT) baked from the pbrt-v3 BSSRDF model.
//  // The 2D Pre-Integrated Skin Look-Up Table (LUT) baked from the pbrt-v3 BSSRDF model.
    // This texture maps the relationship between N.L (dot product of Normal and Light) and surface curvature to pre-calculated diffuse scattering.
//  // This texture maps the relationship between N.L (dot product of Normal and Light) and surface curvature to pre-calculated diffuse scattering.
    // This should be imported in Godot as a 2D Texture with dimensions 512x512.
//  // This should be imported in Godot as a 2D Texture with dimensions 512x512.
    uniform sampler2D pre_integrated_skin_scattering_lut_texture : hint_default_black, filter_linear, repeat_disable;
//  uniform sampler2D pre_integrated_skin_scattering_lut_texture : hint_default_black, filter_linear, repeat_disable;

    // --- Basic Skin Material Properties ---
//  // --- Basic Skin Material Properties ---
    // These properties define the foundational surface color and detail variations of the skin material.
//  // These properties define the foundational surface color and detail variations of the skin material.
    group_uniforms base_color;
//  group_uniforms base_color;
    uniform vec3 skin_base_albedo_color : source_color = vec3(1.0, 1.0, 1.0);
//  uniform vec3 skin_base_albedo_color : source_color = vec3(1.0, 1.0, 1.0);
    // A detail texture that adds pores, freckles, and other skin imperfections to the base color.
//  // A detail texture that adds pores, freckles, and other skin imperfections to the base color.
    uniform sampler2D skin_detail_albedo_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D skin_detail_albedo_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
    // The scale factor used to tile the detail texture across the mesh surface.
//  // The scale factor used to tile the detail texture across the mesh surface.
    uniform vec2 skin_detail_texture_uv_coordinate_scale = vec2(1.0, 1.0);
//  uniform vec2 skin_detail_texture_uv_coordinate_scale = vec2(1.0, 1.0);
    group_uniforms;
//  group_uniforms;

    // --- Micro-Surface Normal Properties ---
//  // --- Micro-Surface Normal Properties ---
    // A normal map texture that adds micro-surface detail without requiring additional geometry.
//  // A normal map texture that adds micro-surface detail without requiring additional geometry.
    group_uniforms normal_map;
//  group_uniforms normal_map;
    uniform sampler2D skin_normal_map_texture : hint_normal, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D skin_normal_map_texture : hint_normal, filter_linear_mipmap, repeat_enable;
    // The scale factor used to tile the normal map texture across the mesh surface.
//  // The scale factor used to tile the normal map texture across the mesh surface.
    uniform vec2 skin_normal_map_texture_uv_coordinate_scale = vec2(1.0, 1.0);
//  uniform vec2 skin_normal_map_texture_uv_coordinate_scale = vec2(1.0, 1.0);
    // The overall strength of the normal map effect.
//  // The overall strength of the normal map effect.
    uniform float skin_normal_map_intensity : hint_range(0.0, 5.0) = 1.0;
//  uniform float skin_normal_map_intensity : hint_range(0.0, 5.0) = 1.0;
    group_uniforms;
//  group_uniforms;

    // --- Specular Reflection Properties ---
//  // --- Specular Reflection Properties ---
    // These properties control the specular reflection layer using the GGX microfacet model.
//  // These properties control the specular reflection layer using the GGX microfacet model.
    group_uniforms specular;
//  group_uniforms specular;
    // The overall strength of the specular reflection on the skin surface.
//  // The overall strength of the specular reflection on the skin surface.
    uniform float skin_direct_specular_intensity : hint_range(0.0, 5.0) = 0.5;
//  uniform float skin_direct_specular_intensity : hint_range(0.0, 5.0) = 0.5;
    // The perceived microscopic roughness of the skin surface, which blurs specular highlights.
//  // The perceived microscopic roughness of the skin surface, which blurs specular highlights.
    uniform float skin_surface_microfacet_roughness : hint_range(0.0, 1.0) = 0.4;
//  uniform float skin_surface_microfacet_roughness : hint_range(0.0, 1.0) = 0.4;
    group_uniforms;
//  group_uniforms;

    // --- Surface Moisture Properties ---
//  // --- Surface Moisture Properties ---
    // These properties control the simulation of surface moisture such as sweat, oil, or water on the skin.
//  // These properties control the simulation of surface moisture such as sweat, oil, or water on the skin.
    group_uniforms surface_moisture;
//  group_uniforms surface_moisture;
    // Toggles the surface moisture simulation for sweat, oil, or water on the skin.
//  // Toggles the surface moisture simulation for sweat, oil, or water on the skin.
    uniform bool enable_skin_surface_moisture = false;
//  uniform bool enable_skin_surface_moisture = false;
    // Controls the overall intensity of the surface moisture effect.
//  // Controls the overall intensity of the surface moisture effect.
    uniform float skin_surface_moisture_intensity : hint_range(0.0, 1.0) = 0.0;
//  uniform float skin_surface_moisture_intensity : hint_range(0.0, 1.0) = 0.0;
    // Controls the perceived microscopic roughness of moisture-affected skin areas.
//  // Controls the perceived microscopic roughness of moisture-affected skin areas.
    uniform float skin_surface_moisture_roughness : hint_range(0.0, 1.0) = 0.15;
//  uniform float skin_surface_moisture_roughness : hint_range(0.0, 1.0) = 0.15;
    // Multiplies the specular reflection intensity in moisture-affected areas to simulate a wet, glossy appearance.
//  // Multiplies the specular reflection intensity in moisture-affected areas to simulate a wet, glossy appearance.
    uniform float skin_surface_moisture_specular_multiplier : hint_range(1.0, 3.0) = 1.5;
//  uniform float skin_surface_moisture_specular_multiplier : hint_range(1.0, 3.0) = 1.5;
    // Controls the amount of albedo darkening applied to moisture-affected skin, simulating light absorption by surface water.
//  // Controls the amount of albedo darkening applied to moisture-affected skin, simulating light absorption by surface water.
    uniform float skin_surface_moisture_albedo_darkening : hint_range(0.0, 1.0) = 0.2;
//  uniform float skin_surface_moisture_albedo_darkening : hint_range(0.0, 1.0) = 0.2;
    // An optional mask texture that spatially defines moisture-affected areas such as sweat patches.
//  // An optional mask texture that spatially defines moisture-affected areas such as sweat patches.
    uniform sampler2D skin_surface_moisture_mask_texture : hint_default_white, filter_linear_mipmap;
//  uniform sampler2D skin_surface_moisture_mask_texture : hint_default_white, filter_linear_mipmap;
    group_uniforms;
//  group_uniforms;

    // --- Subsurface Scattering Properties ---
//  // --- Subsurface Scattering Properties ---
    // These properties control the subsurface scattering effect, which simulates how light bleeds through thin geometry.
//  // These properties control the subsurface scattering effect, which simulates how light bleeds through thin geometry.
    group_uniforms subsurface_scattering;
//  group_uniforms subsurface_scattering;
    // Controls the overall size and spread of the subsurface scattering effect relative to the physical model scale.
//  // Controls the overall size and spread of the subsurface scattering effect relative to the physical model scale.
    uniform float subsurface_scattering_radius_scale : hint_range(0.0, 10.0) = 1.0;
//  uniform float subsurface_scattering_radius_scale : hint_range(0.0, 10.0) = 1.0;
    // The maximum expected curvature of the mesh. This value must exactly match the max_curvature used during the bake_skin.py generation process to map the LUT correctly.
//  // The maximum expected curvature of the mesh. This value must exactly match the max_curvature used during the bake_skin.py generation process to map the LUT correctly.
    uniform float maximum_estimated_surface_curvature : hint_range(0.1, 10.0) = 1.0;
//  uniform float maximum_estimated_surface_curvature : hint_range(0.1, 10.0) = 1.0;
    group_uniforms;
//  group_uniforms;

    // Precomputed constants for the half-texel correction specific to the 512x512 LUT dimensions.
//  // Precomputed constants for the half-texel correction specific to the 512x512 LUT dimensions.
    // These constants prevent sampling artifacts and interpolation errors near the edges of the 2D texture.
//  // These constants prevent sampling artifacts and interpolation errors near the edges of the 2D texture.
    const float constant_skin_lut_half_texel_scale = 0.998046875;
//  const float constant_skin_lut_half_texel_scale = 0.998046875;
    const float constant_skin_lut_half_texel_offset = 0.0009765625;
//  const float constant_skin_lut_half_texel_offset = 0.0009765625;

    // Varyings to pass calculated data from the fragment function to the custom light function.
//  // Varyings to pass calculated data from the fragment function to the custom light function.
    varying vec3 fragment_stored_raw_base_color;
//  varying vec3 fragment_stored_raw_base_color;
    varying float fragment_estimated_surface_curvature;
//  varying float fragment_estimated_surface_curvature;
    varying float fragment_stored_applied_specular_intensity;
//  varying float fragment_stored_applied_specular_intensity;

    void vertex() {
//  void vertex() {
        // Standard vertex transform pass.
//      // Standard vertex transform pass.
        // This is left empty to allow Godot's default vertex processing to handle geometry and skeletal transformations.
//      // This is left empty to allow Godot's default vertex processing to handle geometry and skeletal transformations.
    }
//  }

    void fragment() {
//  void fragment() {
        // Samples the detail texture using the scaled UV coordinates.
//      // Samples the detail texture using the scaled UV coordinates.
        vec4 sampled_skin_detail_texture_color = texture(skin_detail_albedo_texture, UV * skin_detail_texture_uv_coordinate_scale);
//      vec4 sampled_skin_detail_texture_color = texture(skin_detail_albedo_texture, UV * skin_detail_texture_uv_coordinate_scale);
        // Multiplies the uniform base color by the sampled detail texture to produce the final skin albedo base color.
//      // Multiplies the uniform base color by the sampled detail texture to produce the final skin albedo base color.
        vec3 calculated_final_skin_base_color = skin_base_albedo_color * sampled_skin_detail_texture_color.rgb;
//      vec3 calculated_final_skin_base_color = skin_base_albedo_color * sampled_skin_detail_texture_color.rgb;

        // --- Moisture Calculation ---
//      // --- Moisture Calculation ---
        // Initializes the surface properties with their default dry-skin values before moisture adjustments.
//      // Initializes the surface properties with their default dry-skin values before moisture adjustments.
        float applied_skin_surface_microfacet_roughness = skin_surface_microfacet_roughness;
//      float applied_skin_surface_microfacet_roughness = skin_surface_microfacet_roughness;
        float applied_skin_direct_specular_intensity = skin_direct_specular_intensity;
//      float applied_skin_direct_specular_intensity = skin_direct_specular_intensity;

        if (enable_skin_surface_moisture) {
//      if (enable_skin_surface_moisture) {
            // Samples the moisture mask texture to determine the spatial extent of wet areas.
//          // Samples the moisture mask texture to determine the spatial extent of wet areas.
            float sampled_skin_surface_moisture_mask = texture(skin_surface_moisture_mask_texture, UV).r * skin_surface_moisture_intensity;
//          float sampled_skin_surface_moisture_mask = texture(skin_surface_moisture_mask_texture, UV).r * skin_surface_moisture_intensity;

            // Darkens the albedo color based on the moisture mask to simulate light absorption by surface water.
//          // Darkens the albedo color based on the moisture mask to simulate light absorption by surface water.
            calculated_final_skin_base_color = mix(calculated_final_skin_base_color, calculated_final_skin_base_color * (1.0 - skin_surface_moisture_albedo_darkening), sampled_skin_surface_moisture_mask);
//          calculated_final_skin_base_color = mix(calculated_final_skin_base_color, calculated_final_skin_base_color * (1.0 - skin_surface_moisture_albedo_darkening), sampled_skin_surface_moisture_mask);

            // Interpolates between the base roughness and the moisture roughness based on the moisture mask.
//          // Interpolates between the base roughness and the moisture roughness based on the moisture mask.
            applied_skin_surface_microfacet_roughness = mix(skin_surface_microfacet_roughness, skin_surface_moisture_roughness, sampled_skin_surface_moisture_mask);
//          applied_skin_surface_microfacet_roughness = mix(skin_surface_microfacet_roughness, skin_surface_moisture_roughness, sampled_skin_surface_moisture_mask);

            // Boosts the specular intensity in wet areas by interpolating toward the moisture-adjusted specular multiplier.
//          // Boosts the specular intensity in wet areas by interpolating toward the moisture-adjusted specular multiplier.
            applied_skin_direct_specular_intensity = mix(skin_direct_specular_intensity, skin_direct_specular_intensity * skin_surface_moisture_specular_multiplier, sampled_skin_surface_moisture_mask);
//          applied_skin_direct_specular_intensity = mix(skin_direct_specular_intensity, skin_direct_specular_intensity * skin_surface_moisture_specular_multiplier, sampled_skin_surface_moisture_mask);
        }
//      }

        // Pass the raw calculated base color to the light function via a varying variable.
//      // Pass the raw calculated base color to the light function via a varying variable.
        // This optimization avoids redundant texture sampling during the per-pixel/per-light evaluation in the light() pass.
//      // This optimization avoids redundant texture sampling during the per-pixel/per-light evaluation in the light() pass.
        fragment_stored_raw_base_color = calculated_final_skin_base_color;
//      fragment_stored_raw_base_color = calculated_final_skin_base_color;

        // Pass the moisture-adjusted specular intensity to the light function via a varying variable.
//      // Pass the moisture-adjusted specular intensity to the light function via a varying variable.
        // This ensures the custom GGX specular calculation in light() respects the surface moisture adjustment from fragment().
//      // This ensures the custom GGX specular calculation in light() respects the surface moisture adjustment from fragment().
        fragment_stored_applied_specular_intensity = applied_skin_direct_specular_intensity;
//      fragment_stored_applied_specular_intensity = applied_skin_direct_specular_intensity;

        // Estimate local geometric curvature dynamically using screen-space partial derivatives.
//      // Estimate local geometric curvature dynamically using screen-space partial derivatives.
        // The fwidth function gives the sum of absolute derivatives with respect to screen x and y axes.
//      // The fwidth function gives the sum of absolute derivatives with respect to screen x and y axes.
        // We measure how much the normal changes relative to how much the position changes across adjacent pixels.
//      // We measure how much the normal changes relative to how much the position changes across adjacent pixels.
        float screen_space_derivative_of_surface_normal = length(fwidth(NORMAL));
//      float screen_space_derivative_of_surface_normal = length(fwidth(NORMAL));
        float screen_space_derivative_of_surface_position = length(fwidth(VERTEX));
//      float screen_space_derivative_of_surface_position = length(fwidth(VERTEX));

        // Calculate Curvature (C) as the rate of change of the normal (dN) over the change in surface position (ds).
//      // Calculate Curvature (C) as the rate of change of the normal (dN) over the change in surface position (ds).
        // A small epsilon (0.0001) is added to the denominator to prevent division by zero artifacts on flat surfaces.
//      // A small epsilon (0.0001) is added to the denominator to prevent division by zero artifacts on flat surfaces.
        fragment_estimated_surface_curvature = screen_space_derivative_of_surface_normal / (screen_space_derivative_of_surface_position + 0.0001);
//      fragment_estimated_surface_curvature = screen_space_derivative_of_surface_normal / (screen_space_derivative_of_surface_position + 0.0001);

        // Set the built-in PBR properties to properly interact with Godot's ambient, indirect lighting, and reflection probes.
//      // Set the built-in PBR properties to properly interact with Godot's ambient, indirect lighting, and reflection probes.
        ALBEDO = calculated_final_skin_base_color;
//      ALBEDO = calculated_final_skin_base_color;

        // Applies the calculated roughness, which may be adjusted by the surface moisture simulation.
//      // Applies the calculated roughness, which may be adjusted by the surface moisture simulation.
        ROUGHNESS = applied_skin_surface_microfacet_roughness;
//      ROUGHNESS = applied_skin_surface_microfacet_roughness;

        METALLIC = 0.0; // Skin is a dielectric material, so metallic is strictly 0.0.
//      METALLIC = 0.0; // Skin is a dielectric material, so metallic is strictly 0.0.

        // Applies the calculated specular intensity, which may be boosted by the surface moisture simulation.
//      // Applies the calculated specular intensity, which may be boosted by the surface moisture simulation.
        SPECULAR = applied_skin_direct_specular_intensity;
//      SPECULAR = applied_skin_direct_specular_intensity;

        // Samples the normal map texture using the scaled UV coordinates.
//      // Samples the normal map texture using the scaled UV coordinates.
        vec3 sampled_skin_normal_map_color = texture(skin_normal_map_texture, UV * skin_normal_map_texture_uv_coordinate_scale).rgb;
//      vec3 sampled_skin_normal_map_color = texture(skin_normal_map_texture, UV * skin_normal_map_texture_uv_coordinate_scale).rgb;

        // Applies the sampled normal map to the built-in PBR normal map property.
//      // Applies the sampled normal map to the built-in PBR normal map property.
        NORMAL_MAP = sampled_skin_normal_map_color;
//      NORMAL_MAP = sampled_skin_normal_map_color;

        // Scales the overall intensity of the normal map effect.
//      // Scales the overall intensity of the normal map effect.
        NORMAL_MAP_DEPTH = skin_normal_map_intensity;
//      NORMAL_MAP_DEPTH = skin_normal_map_intensity;
    }
//  }

    void light() {
//  void light() {
        // Normalize the core vectors required for lighting calculations to ensure accurate dot products.
//      // Normalize the core vectors required for lighting calculations to ensure accurate dot products.
        vec3 normalized_incident_light_direction = normalize(LIGHT);
//      vec3 normalized_incident_light_direction = normalize(LIGHT);
        vec3 normalized_surface_normal_vector = normalize(NORMAL);
//      vec3 normalized_surface_normal_vector = normalize(NORMAL);
        vec3 normalized_outgoing_view_direction = normalize(VIEW);
//      vec3 normalized_outgoing_view_direction = normalize(VIEW);

        // Calculate the cosine of the angle between the surface normal and the incident light direction.
//      // Calculate the cosine of the angle between the surface normal and the incident light direction.
        // This determines how directly the light strikes the surface (-1.0 to 1.0).
//      // This determines how directly the light strikes the surface (-1.0 to 1.0).
        float dot_product_normal_and_light_direction = dot(normalized_surface_normal_vector, normalized_incident_light_direction);
//      float dot_product_normal_and_light_direction = dot(normalized_surface_normal_vector, normalized_incident_light_direction);

        // --- Subsurface Scattering Diffuse Evaluation (Pre-Integrated Skin Approach) ---
//      // --- Subsurface Scattering Diffuse Evaluation (Pre-Integrated Skin Approach) ---

        // Map the N.L dot product from its original range [-1, 1] to a normalized [0, 1] range.
//      // Map the N.L dot product from its original range [-1, 1] to a normalized [0, 1] range.
        // This value is used as the horizontal (U) coordinate for sampling the pre-integrated look-up texture.
//      // This value is used as the horizontal (U) coordinate for sampling the pre-integrated look-up texture.
        float lut_texture_coordinate_u_from_n_dot_l = dot_product_normal_and_light_direction * 0.5 + 0.5;
//      float lut_texture_coordinate_u_from_n_dot_l = dot_product_normal_and_light_direction * 0.5 + 0.5;

        // Map the estimated dynamic curvature into the vertical (V) coordinate for the LUT.
//      // Map the estimated dynamic curvature into the vertical (V) coordinate for the LUT.
        // We scale the curvature by the user-defined scattering scale and divide by the baked maximum curvature limit.
//      // We scale the curvature by the user-defined scattering scale and divide by the baked maximum curvature limit.
        // The clamp function ensures the V coordinate remains strictly within the valid [0, 1] texture boundary.
//      // The clamp function ensures the V coordinate remains strictly within the valid [0, 1] texture boundary.
        float lut_texture_coordinate_v_from_curvature = clamp((fragment_estimated_surface_curvature * subsurface_scattering_radius_scale) / maximum_estimated_surface_curvature, 0.0, 1.0);
//      float lut_texture_coordinate_v_from_curvature = clamp((fragment_estimated_surface_curvature * subsurface_scattering_radius_scale) / maximum_estimated_surface_curvature, 0.0, 1.0);

        // Apply a half-texel offset and scale correction to ensure proper texel-center sampling at LUT boundaries.
//      // Apply a half-texel offset and scale correction to ensure proper texel-center sampling at LUT boundaries.
        // For 512-texel dimensions: scale = 511.0 / 512.0 = 0.998046875, offset = 0.5 / 512.0 = 0.0009765625.
//      // For 512-texel dimensions: scale = 511.0 / 512.0 = 0.998046875, offset = 0.5 / 512.0 = 0.0009765625.
        lut_texture_coordinate_u_from_n_dot_l = lut_texture_coordinate_u_from_n_dot_l * constant_skin_lut_half_texel_scale + constant_skin_lut_half_texel_offset;
//      lut_texture_coordinate_u_from_n_dot_l = lut_texture_coordinate_u_from_n_dot_l * constant_skin_lut_half_texel_scale + constant_skin_lut_half_texel_offset;
        lut_texture_coordinate_v_from_curvature = lut_texture_coordinate_v_from_curvature * constant_skin_lut_half_texel_scale + constant_skin_lut_half_texel_offset;
//      lut_texture_coordinate_v_from_curvature = lut_texture_coordinate_v_from_curvature * constant_skin_lut_half_texel_scale + constant_skin_lut_half_texel_offset;

        // Sample the Pre-Integrated Skin LUT using the calculated U (N.L) and V (Curvature) coordinates.
//      // Sample the Pre-Integrated Skin LUT using the calculated U (N.L) and V (Curvature) coordinates.
        // This yields the blurred, scattered diffuse light profile for human skin.
//      // This yields the blurred, scattered diffuse light profile for human skin.
        vec3 sampled_pre_integrated_subsurface_scattering_diffuse_color = texture(pre_integrated_skin_scattering_lut_texture, vec2(lut_texture_coordinate_u_from_n_dot_l, lut_texture_coordinate_v_from_curvature)).rgb;
//      vec3 sampled_pre_integrated_subsurface_scattering_diffuse_color = texture(pre_integrated_skin_scattering_lut_texture, vec2(lut_texture_coordinate_u_from_n_dot_l, lut_texture_coordinate_v_from_curvature)).rgb;

        // Modulate the scattered white light profile with the actual unlit base color of the skin.
//      // Modulate the scattered white light profile with the actual unlit base color of the skin.
        vec3 calculated_final_diffuse_subsurface_scattering_light = sampled_pre_integrated_subsurface_scattering_diffuse_color * fragment_stored_raw_base_color;
//      vec3 calculated_final_diffuse_subsurface_scattering_light = sampled_pre_integrated_subsurface_scattering_diffuse_color * fragment_stored_raw_base_color;

        // Integrate the final subsurface scattering color into the DIFFUSE_LIGHT output.
//      // Integrate the final subsurface scattering color into the DIFFUSE_LIGHT output.
        // We multiply by the incoming LIGHT_COLOR and the structural ATTENUATION (shadows, light falloff).
//      // We multiply by the incoming LIGHT_COLOR and the structural ATTENUATION (shadows, light falloff).
        DIFFUSE_LIGHT += calculated_final_diffuse_subsurface_scattering_light * LIGHT_COLOR * ATTENUATION;
//      DIFFUSE_LIGHT += calculated_final_diffuse_subsurface_scattering_light * LIGHT_COLOR * ATTENUATION;

        // --- Direct Specular Reflection Evaluation (GGX Microfacet Model) ---
//      // --- Direct Specular Reflection Evaluation (GGX Microfacet Model) ---

        // Calculate the half-vector, which is exactly halfway between the light direction and the view direction.
//      // Calculate the half-vector, which is exactly halfway between the light direction and the view direction.
        vec3 calculated_microfacet_half_vector = normalize(normalized_incident_light_direction + normalized_outgoing_view_direction);
//      vec3 calculated_microfacet_half_vector = normalize(normalized_incident_light_direction + normalized_outgoing_view_direction);

        // Calculate required dot products for the GGX specular BRDF equation.
//      // Calculate required dot products for the GGX specular BRDF equation.
        // Max functions ensure these dot products never drop below 0.0, which would cause physically impossible negative lighting.
//      // Max functions ensure these dot products never drop below 0.0, which would cause physically impossible negative lighting.
        float dot_product_normal_and_half_vector = max(0.0, dot(normalized_surface_normal_vector, calculated_microfacet_half_vector));
//      float dot_product_normal_and_half_vector = max(0.0, dot(normalized_surface_normal_vector, calculated_microfacet_half_vector));
        float dot_product_normal_and_view_direction = max(0.0, dot(normalized_surface_normal_vector, normalized_outgoing_view_direction));
//      float dot_product_normal_and_view_direction = max(0.0, dot(normalized_surface_normal_vector, normalized_outgoing_view_direction));

        // 1. Normal Distribution Function (D) - Trowbridge-Reitz GGX
//      // 1. Normal Distribution Function (D) - Trowbridge-Reitz GGX
        // This function determines the statistical distribution of microfacet normals aligned with the half-vector.
//      // This function determines the statistical distribution of microfacet normals aligned with the half-vector.
        float microfacet_roughness_alpha = ROUGHNESS * ROUGHNESS;
//      float microfacet_roughness_alpha = ROUGHNESS * ROUGHNESS;
        float microfacet_roughness_alpha_squared = microfacet_roughness_alpha * microfacet_roughness_alpha;
//      float microfacet_roughness_alpha_squared = microfacet_roughness_alpha * microfacet_roughness_alpha;
        float ggx_distribution_denominator = dot_product_normal_and_half_vector * dot_product_normal_and_half_vector * (microfacet_roughness_alpha_squared - 1.0) + 1.0;
//      float ggx_distribution_denominator = dot_product_normal_and_half_vector * dot_product_normal_and_half_vector * (microfacet_roughness_alpha_squared - 1.0) + 1.0;
        float ggx_normal_distribution_function_d = microfacet_roughness_alpha_squared / (PI * ggx_distribution_denominator * ggx_distribution_denominator);
//      float ggx_normal_distribution_function_d = microfacet_roughness_alpha_squared / (PI * ggx_distribution_denominator * ggx_distribution_denominator);

        // 2. Fresnel Equation (F) - Schlick's Approximation
//      // 2. Fresnel Equation (F) - Schlick's Approximation
        // This describes how the ratio of reflected vs. refracted light changes based on the viewing angle.
//      // This describes how the ratio of reflected vs. refracted light changes based on the viewing angle.
        // Base reflectivity (F0) for biological tissue like human skin is typically around 0.028 (assuming an Index of Refraction of ~1.4).
//      // Base reflectivity (F0) for biological tissue like human skin is typically around 0.028 (assuming an Index of Refraction of ~1.4).
        vec3 fresnel_reflectance_at_normal_incidence_f0 = vec3(0.028);
//      vec3 fresnel_reflectance_at_normal_incidence_f0 = vec3(0.028);
        vec3 schlick_fresnel_approximation_f = fresnel_reflectance_at_normal_incidence_f0 + (1.0 - fresnel_reflectance_at_normal_incidence_f0) * pow(1.0 - max(0.0, dot(calculated_microfacet_half_vector, normalized_outgoing_view_direction)), 5.0);
//      vec3 schlick_fresnel_approximation_f = fresnel_reflectance_at_normal_incidence_f0 + (1.0 - fresnel_reflectance_at_normal_incidence_f0) * pow(1.0 - max(0.0, dot(calculated_microfacet_half_vector, normalized_outgoing_view_direction)), 5.0);

        // 3. Geometry Function (G) - Smith's method with Schlick-GGX approximation
//      // 3. Geometry Function (G) - Smith's method with Schlick-GGX approximation
        // This function accounts for self-shadowing and self-masking of the microfacets, preventing them from reflecting too much light.
//      // This function accounts for self-shadowing and self-masking of the microfacets, preventing them from reflecting too much light.
        float schlick_smith_geometry_roughness_factor_k = (ROUGHNESS + 1.0);
//      float schlick_smith_geometry_roughness_factor_k = (ROUGHNESS + 1.0);
        schlick_smith_geometry_roughness_factor_k = (schlick_smith_geometry_roughness_factor_k * schlick_smith_geometry_roughness_factor_k) / 8.0;
//      schlick_smith_geometry_roughness_factor_k = (schlick_smith_geometry_roughness_factor_k * schlick_smith_geometry_roughness_factor_k) / 8.0;

        float smith_geometry_shadowing_function_light_g_l = max(0.0, dot_product_normal_and_light_direction) / (max(0.0, dot_product_normal_and_light_direction) * (1.0 - schlick_smith_geometry_roughness_factor_k) + schlick_smith_geometry_roughness_factor_k);
//      float smith_geometry_shadowing_function_light_g_l = max(0.0, dot_product_normal_and_light_direction) / (max(0.0, dot_product_normal_and_light_direction) * (1.0 - schlick_smith_geometry_roughness_factor_k) + schlick_smith_geometry_roughness_factor_k);
        float smith_geometry_masking_function_view_g_v = dot_product_normal_and_view_direction / (dot_product_normal_and_view_direction * (1.0 - schlick_smith_geometry_roughness_factor_k) + schlick_smith_geometry_roughness_factor_k);
//      float smith_geometry_masking_function_view_g_v = dot_product_normal_and_view_direction / (dot_product_normal_and_view_direction * (1.0 - schlick_smith_geometry_roughness_factor_k) + schlick_smith_geometry_roughness_factor_k);
        float smith_joint_geometry_function_g = smith_geometry_shadowing_function_light_g_l * smith_geometry_masking_function_view_g_v;
//      float smith_joint_geometry_function_g = smith_geometry_shadowing_function_light_g_l * smith_geometry_masking_function_view_g_v;

        // Combine D, F, and G into the final Cook-Torrance Specular BRDF equation.
//      // Combine D, F, and G into the final Cook-Torrance Specular BRDF equation.
        // We divide by a normalizing factor to conserve energy, adding a tiny epsilon to prevent division by zero.
//      // We divide by a normalizing factor to conserve energy, adding a tiny epsilon to prevent division by zero.
        vec3 calculated_ggx_specular_brdf_reflection = (ggx_normal_distribution_function_d * schlick_fresnel_approximation_f * smith_joint_geometry_function_g) / (4.0 * dot_product_normal_and_view_direction + 0.0001);
//      vec3 calculated_ggx_specular_brdf_reflection = (ggx_normal_distribution_function_d * schlick_fresnel_approximation_f * smith_joint_geometry_function_g) / (4.0 * dot_product_normal_and_view_direction + 0.0001);

        // Apply specular intensity scaling and completely mask out specular highlights on surfaces facing away from the light.
//      // Apply specular intensity scaling and completely mask out specular highlights on surfaces facing away from the light.
        // The sign(ndotl) evaluates to -1 on backfaces, which max(0.0, ...) clips to 0, ensuring shadows don't incorrectly generate specular reflections.
//      // The sign(ndotl) evaluates to -1 on backfaces, which max(0.0, ...) clips to 0, ensuring shadows don't incorrectly generate specular reflections.
        float shadow_masked_specular_intensity = max(0.0, sign(dot_product_normal_and_light_direction)) * fragment_stored_applied_specular_intensity;
//      float shadow_masked_specular_intensity = max(0.0, sign(dot_product_normal_and_light_direction)) * fragment_stored_applied_specular_intensity;

        // Integrate the final specular reflection into the SPECULAR_LIGHT output.
//      // Integrate the final specular reflection into the SPECULAR_LIGHT output.
        // Multiply by LIGHT_COLOR, ATTENUATION, and our computed intensity mask.
//      // Multiply by LIGHT_COLOR, ATTENUATION, and our computed intensity mask.
        SPECULAR_LIGHT += calculated_ggx_specular_brdf_reflection * LIGHT_COLOR * ATTENUATION * shadow_masked_specular_intensity;
//      SPECULAR_LIGHT += calculated_ggx_specular_brdf_reflection * LIGHT_COLOR * ATTENUATION * shadow_masked_specular_intensity;
    }
//  }
