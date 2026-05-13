    shader_type spatial;
//  shader_type spatial;
    render_mode blend_mix, depth_draw_opaque, depth_prepass_alpha, cull_disabled, specular_disabled;
//  render_mode blend_mix, depth_draw_opaque, depth_prepass_alpha, cull_disabled, specular_disabled;

    // A 3D texture containing the baked PBRT hair Bidirectional Scattering Distribution Function (BSDF).
//  // A 3D texture containing the baked PBRT hair Bidirectional Scattering Distribution Function (BSDF).
    // This Look-Up Table (LUT) stores the precomputed complex hair scattering properties.
//  // This Look-Up Table (LUT) stores the precomputed complex hair scattering properties.
    // It must be imported into Godot as a 3D texture with dimensions of 64x64x64 (8 columns, 8 rows).
//  // It must be imported into Godot as a 3D texture with dimensions of 64x64x64 (8 columns, 8 rows).
    uniform sampler3D hair_bsdf_texture_lut : hint_default_black, filter_linear, repeat_disable;
//  uniform sampler3D hair_bsdf_texture_lut : hint_default_black, filter_linear, repeat_disable;

    // --- Two-Pass Transparency Setup & Documentation ---
//  // --- Two-Pass Transparency Setup & Documentation ---
    // Mechanism & Algorithm:
//  // Mechanism & Algorithm:
    // Real-time rasterizers struggle to sort complex overlapping transparent geometry (such as hair strands) properly.
//  // Real-time rasterizers struggle to sort complex overlapping transparent geometry (such as hair strands) properly.
    // The two-pass system solves this issue by breaking the rendering process into two distinct stages:
//  // The two-pass system solves this issue by breaking the rendering process into two distinct stages:
    // 1. Pass 1 (Opaque Core): Renders only the dense, opaque parts of the hair (where Alpha >= Threshold) to the depth buffer.
//  // 1. Pass 1 (Opaque Core): Renders only the dense, opaque parts of the hair (where Alpha >= Threshold) to the depth buffer.
    //    This pass discards the soft, wispy edges. By writing to the depth buffer, it perfectly occludes the geometry behind it.
//  //    This pass discards the soft, wispy edges. By writing to the depth buffer, it perfectly occludes the geometry behind it.
    // 2. Pass 2 (Soft Fringes): Renders only the soft, transparent edges (where Alpha < Threshold) over the top.
//  // 2. Pass 2 (Soft Fringes): Renders only the soft, transparent edges (where Alpha < Threshold) over the top.
    //    This pass discards the dense core to prevent overdraw. Because Pass 1 has already written to the depth buffer, these transparent edges
//  //    This pass discards the dense core to prevent overdraw. Because Pass 1 has already written to the depth buffer, these transparent edges
    //    will sort perfectly against the underlying solid hair structures.
//  //    will sort perfectly against the underlying solid hair structures.
    //
//  //
    // How to set this up in Godot:
//  // How to set this up in Godot:
    // Option A: Mode 0 (Single Pass Soft)
//  // Option A: Mode 0 (Single Pass Soft)
    // - Use this mode if perfect depth sorting is unnecessary, and you only require soft blending.
//  // - Use this mode if perfect depth sorting is unnecessary, and you only require soft blending.
    // - WARNING: Mode 0 has inherent transparency sorting artifacts with overlapping hair geometry
//  // - WARNING: Mode 0 has inherent transparency sorting artifacts with overlapping hair geometry
    // -   due to depth_prepass_alpha writing depth for semi-transparent fragments.
//  // -   due to depth_prepass_alpha writing depth for semi-transparent fragments.
    // -   For correct sorting, use Option B (Mode 1 + Mode 2) instead.
//  // -   For correct sorting, use Option B (Mode 1 + Mode 2) instead.
    // - Set render_pass_mode to 0 on your primary material.
//  // - Set render_pass_mode to 0 on your primary material.
    //
//  //
    // Option B: Mode 1 + Mode 2 (Perfect Two-Pass Sorting)
//  // Option B: Mode 1 + Mode 2 (Perfect Two-Pass Sorting)
    // - 1. Apply a material with this shader to your mesh. Set render_pass_mode to 1, and set opaque_core_threshold to approximately 0.5.
//  // - 1. Apply a material with this shader to your mesh. Set render_pass_mode to 1, and set opaque_core_threshold to approximately 0.5.
    // - 2. Navigate to the "Next Pass" property of that material and add a new ShaderMaterial utilizing this exact same shader.
//  // - 2. Navigate to the "Next Pass" property of that material and add a new ShaderMaterial utilizing this exact same shader.
    // - 3. On the Next Pass material, set render_pass_mode to 2. Ensure the opaque_core_threshold matches the value from the first material.
//  // - 3. On the Next Pass material, set render_pass_mode to 2. Ensure the opaque_core_threshold matches the value from the first material.
    // ---------------------------------------------------
//  // ---------------------------------------------------

    // 0 = Single Pass Soft (Original), 1 = Pass 1 (Opaque Core), 2 = Pass 2 (Soft Fringes).
//  // 0 = Single Pass Soft (Original), 1 = Pass 1 (Opaque Core), 2 = Pass 2 (Soft Fringes).
    group_uniforms transparency;
//  group_uniforms transparency;
    uniform int render_pass_mode : hint_range(0, 2, 1) = 0;
//  uniform int render_pass_mode : hint_range(0, 2, 1) = 0;
    // Specifies the alpha threshold for the opaque core pass (Pass 1).
//  // Specifies the alpha threshold for the opaque core pass (Pass 1).
    uniform float opaque_core_threshold : hint_range(0.0, 1.0) = 0.5;
//  uniform float opaque_core_threshold : hint_range(0.0, 1.0) = 0.5;
    group_uniforms;
//  group_uniforms;

    // Basic hair material properties.
//  // Basic hair material properties.
    // These properties define the foundational visual characteristics of the hair surface.
//  // These properties define the foundational visual characteristics of the hair surface.
    group_uniforms base_color;
//  group_uniforms base_color;
    uniform vec4 base_albedo_color : source_color = vec4(1.0, 1.0, 1.0, 1.0);
//  uniform vec4 base_albedo_color : source_color = vec4(1.0, 1.0, 1.0, 1.0);
    uniform sampler2D detail_albedo_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D detail_albedo_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
    uniform vec2 detail_texture_uv_scale = vec2(1.0, 1.0);
//  uniform vec2 detail_texture_uv_scale = vec2(1.0, 1.0);
    uniform float direct_bsdf_intensity : hint_range(0.0, 5.0) = 0.3;
//  uniform float direct_bsdf_intensity : hint_range(0.0, 5.0) = 0.3;
    // The sharpness of the analytical anisotropic highlight.
//  // The sharpness of the analytical anisotropic highlight.
    uniform float hair_specular_power : hint_range(1.0, 128.0) = 60.0;
//  uniform float hair_specular_power : hint_range(1.0, 128.0) = 60.0;
    group_uniforms;
//  group_uniforms;

    // A normal map texture that adds micro-surface details without requiring additional geometry.
//  // A normal map texture that adds micro-surface details without requiring additional geometry.
    group_uniforms normal_map;
//  group_uniforms normal_map;
    uniform sampler2D hair_normal_map_texture : hint_normal, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D hair_normal_map_texture : hint_normal, filter_linear_mipmap, repeat_enable;
    // A scale factor used to tile the normal map texture across the mesh surface.
//  // A scale factor used to tile the normal map texture across the mesh surface.
    uniform vec2 hair_normal_map_texture_uv_coordinate_scale = vec2(1.0, 1.0);
//  uniform vec2 hair_normal_map_texture_uv_coordinate_scale = vec2(1.0, 1.0);
    // The overall strength of the normal map effect.
//  // The overall strength of the normal map effect.
    uniform float hair_normal_map_intensity : hint_range(0.0, 5.0) = 1.0;
//  uniform float hair_normal_map_intensity : hint_range(0.0, 5.0) = 1.0;
    group_uniforms;
//  group_uniforms;

    // Ambient occlusion properties.
//  // Ambient occlusion properties.
    // These properties add localized shadowing to ground the hair strands and simulate self-occlusion.
//  // These properties add localized shadowing to ground the hair strands and simulate self-occlusion.
    group_uniforms ambient_occlusion;
//  group_uniforms ambient_occlusion;
    uniform bool enable_hair_ambient_occlusion = false;
//  uniform bool enable_hair_ambient_occlusion = false;
    uniform sampler2D hair_ambient_occlusion_texture : hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D hair_ambient_occlusion_texture : hint_default_white, filter_linear_mipmap, repeat_enable;
    uniform float hair_ambient_occlusion_intensity : hint_range(0.0, 1.0) = 1.0;
//  uniform float hair_ambient_occlusion_intensity : hint_range(0.0, 1.0) = 1.0;
    uniform vec2 hair_ambient_occlusion_uv_coordinate_scale = vec2(1.0, 1.0);
//  uniform vec2 hair_ambient_occlusion_uv_coordinate_scale = vec2(1.0, 1.0);
    group_uniforms;
//  group_uniforms;

    // Advanced lighting and shading properties.
//  // Advanced lighting and shading properties.
    // These properties control how the hair reacts to direct light sources and indirect ambient lighting.
//  // These properties control how the hair reacts to direct light sources and indirect ambient lighting.
    group_uniforms lighting;
//  group_uniforms lighting;
    uniform float indirect_ambient_albedo_scale : hint_range(0.0, 1.0) = 1.0;
//  uniform float indirect_ambient_albedo_scale : hint_range(0.0, 1.0) = 1.0;
    uniform float direct_light_intensity_scale : hint_range(0.0, 100.0) = 100.0;
//  uniform float direct_light_intensity_scale : hint_range(0.0, 100.0) = 100.0;
    // Controls the intensity of light passing through the hair volume.
//  // Controls the intensity of light passing through the hair volume.
    uniform float backlight_transmission_intensity : hint_range(0.0, 2.0) = 1.0;
//  uniform float backlight_transmission_intensity : hint_range(0.0, 2.0) = 1.0;
    uniform bool enable_custom_ambient_lighting = false;
//  uniform bool enable_custom_ambient_lighting = false;
    uniform vec3 custom_ambient_color : source_color = vec3(0.05, 0.05, 0.05);
//  uniform vec3 custom_ambient_color : source_color = vec3(0.05, 0.05, 0.05);
    group_uniforms;
//  group_uniforms;

    // Post-processing and stylization properties.
//  // Post-processing and stylization properties.
    // This property toggles the film grain noise overlay.
//  // This property toggles the film grain noise overlay.
    group_uniforms post_processing;
//  group_uniforms post_processing;
    uniform bool enable_film_grain_noise = false;
//  uniform bool enable_film_grain_noise = false;
    // This property defines the intensity of stylistic overlays, such as the film grain noise.
//  // This property defines the intensity of stylistic overlays, such as the film grain noise.
    uniform float film_grain_intensity : hint_range(0.0, 1.0) = 0.01;
//  uniform float film_grain_intensity : hint_range(0.0, 1.0) = 0.01;
    // Controls the animation speed of the film grain noise pattern.
//  // Controls the animation speed of the film grain noise pattern.
    uniform vec2 film_grain_noise_speed = vec2(0.0, 0.0);
//  uniform vec2 film_grain_noise_speed = vec2(0.0, 0.0);
    // Controls the animation speed of the dithered alpha threshold noise pattern.
//  // Controls the animation speed of the dithered alpha threshold noise pattern.
    uniform vec2 dither_noise_speed = vec2(0.0, 0.0);
//  uniform vec2 dither_noise_speed = vec2(0.0, 0.0);
    group_uniforms;
//  group_uniforms;

    const float constant_inverse_pi = 0.31830988618;
//  const float constant_inverse_pi = 0.31830988618;
    const float constant_lut_half_texel_scale = 0.984375;
//  const float constant_lut_half_texel_scale = 0.984375;
    const float constant_lut_half_texel_offset = 0.0078125;
//  const float constant_lut_half_texel_offset = 0.0078125;
    const vec3 constant_luminance_weights = vec3(0.299, 0.587, 0.114);
//  const vec3 constant_luminance_weights = vec3(0.299, 0.587, 0.114);

    varying vec3 stored_hair_tangent;
//  varying vec3 stored_hair_tangent;
    varying vec3 stored_raw_base_color;
//  varying vec3 stored_raw_base_color;
    varying float stored_detail_luminance;
//  varying float stored_detail_luminance;

    varying vec3 stored_outgoing_data;
//  varying vec3 stored_outgoing_data;

    varying vec2 scaled_detail_uv;
//  varying vec2 scaled_detail_uv;
    varying vec2 scaled_ao_uv;
//  varying vec2 scaled_ao_uv;
    varying vec2 scaled_normal_uv;
//  varying vec2 scaled_normal_uv;

    // A custom function for calculating ambient and indirect lighting.
//  // A custom function for calculating ambient and indirect lighting.
    // This function allows developers to inject specific logic for global illumination or stylistic ambient shading.
//  // This function allows developers to inject specific logic for global illumination or stylistic ambient shading.
    vec3 compute_custom_ambient_lighting(vec3 surface_base_color_rgb, vec3 surface_normal, vec3 surface_tangent, vec3 view_direction) {
//  vec3 compute_custom_ambient_lighting(vec3 surface_base_color_rgb, vec3 surface_normal, vec3 surface_tangent, vec3 view_direction) {
        // Override this function to implement custom ambient lighting (e.g., hemispheric light, cubemap sample).
//      // Override this function to implement custom ambient lighting (e.g., hemispheric light, cubemap sample).
        return surface_base_color_rgb * custom_ambient_color;
//      return surface_base_color_rgb * custom_ambient_color;
    }
//  }

    // Generates a pseudo-random value based on a 2D input coordinate.
//  // Generates a pseudo-random value based on a 2D input coordinate.
    // This function is utilized to synthesize a static noise pattern for the film grain visual effect.
//  // This function is utilized to synthesize a static noise pattern for the film grain visual effect.
    float compute_random_grain_noise(vec2 input_screen_coordinate) {
//  float compute_random_grain_noise(vec2 input_screen_coordinate) {
        // Computes a pseudo-random float utilizing a dot product and a sine function hash.
//      // Computes a pseudo-random float utilizing a dot product and a sine function hash.
        return fract(sin(dot(input_screen_coordinate, vec2(12.9898, 78.233))) * 43758.5453);
//      return fract(sin(dot(input_screen_coordinate, vec2(12.9898, 78.233))) * 43758.5453);
    }
//  }

    void vertex() {
//  void vertex() {
        // The standard vertex transformation pass.
//      // The standard vertex transformation pass.
        // This function is intentionally left empty to allow Godot's default vertex processing to handle geometry transformations.
//      // This function is intentionally left empty to allow Godot's default vertex processing to handle geometry transformations.
        scaled_detail_uv = UV * detail_texture_uv_scale;
//      scaled_detail_uv = UV * detail_texture_uv_scale;
        scaled_ao_uv = UV * hair_ambient_occlusion_uv_coordinate_scale;
//      scaled_ao_uv = UV * hair_ambient_occlusion_uv_coordinate_scale;
        scaled_normal_uv = UV * hair_normal_map_texture_uv_coordinate_scale;
//      scaled_normal_uv = UV * hair_normal_map_texture_uv_coordinate_scale;
    }
//  }

    void fragment() {
//  void fragment() {
        // --- Early Alpha Calculation & Discard ---
//      // --- Early Alpha Calculation & Discard ---
        // Samples the full detail texel once. The alpha is used for early discard; RGB is used later for color.
//      // Samples the full detail texel once. The alpha is used for early discard; RGB is used later for color.
        vec4 sampled_detail_texel = texture(detail_albedo_texture, scaled_detail_uv);
//      vec4 sampled_detail_texel = texture(detail_albedo_texture, scaled_detail_uv);
        float sampled_detail_alpha = sampled_detail_texel.a;
//      float sampled_detail_alpha = sampled_detail_texel.a;

        // Applies the alpha from the detail texture and the base color to determine the hair's transparency and sorting.
//      // Applies the alpha from the detail texture and the base color to determine the hair's transparency and sorting.
        ALPHA = base_albedo_color.a * sampled_detail_alpha;
//      ALPHA = base_albedo_color.a * sampled_detail_alpha;

        // --- Two-Pass Transparency Logic ---
//      // --- Two-Pass Transparency Logic ---
        // Generates a high-frequency dither noise value to soften the clipping threshold for fringes.
//      // Generates a high-frequency dither noise value to soften the clipping threshold for fringes.
        float generated_dither_noise_value = compute_random_grain_noise(SCREEN_UV * 100.0 + vec2(TIME) * dither_noise_speed);
//      float generated_dither_noise_value = compute_random_grain_noise(SCREEN_UV * 100.0 + vec2(TIME) * dither_noise_speed);
        float calculated_dithered_threshold = opaque_core_threshold * (0.8 + (generated_dither_noise_value * 0.4));
//      float calculated_dithered_threshold = opaque_core_threshold * (0.8 + (generated_dither_noise_value * 0.4));

        if (render_pass_mode == 1) {
//      if (render_pass_mode == 1) {
            // Pass 1: Opaque Core. Discards soft fringes to write only the solid core to the depth buffer.
//          // Pass 1: Opaque Core. Discards soft fringes to write only the solid core to the depth buffer.
            if (ALPHA < calculated_dithered_threshold) {
//          if (ALPHA < calculated_dithered_threshold) {
                discard;
//              discard;
            }
//          }
            // Forces full opacity for the core to ensure perfect depth writing and shadow casting.
//          // Forces full opacity for the core to ensure perfect depth writing and shadow casting.
            ALPHA = 1.0;
//          ALPHA = 1.0;
        } else if (render_pass_mode == 2) {
//      } else if (render_pass_mode == 2) {
            // Pass 2: Soft Fringes. Renders only the soft, transparent areas, discarding the core to avoid overdraw.
//          // Pass 2: Soft Fringes. Renders only the soft, transparent areas, discarding the core to avoid overdraw.
            if (ALPHA >= calculated_dithered_threshold) {
//          if (ALPHA >= calculated_dithered_threshold) {
                discard;
//              discard;
            }
//          }
        }
//      }

        // --- Expensive Calculations Follow ---
//      // --- Expensive Calculations Follow ---
        // If the fragment was not discarded, the shader proceeds with heavy mathematical operations.
//      // If the fragment was not discarded, the shader proceeds with heavy mathematical operations.
        vec3 sampled_detail_color = sampled_detail_texel.rgb;
//      vec3 sampled_detail_color = sampled_detail_texel.rgb;
        vec3 calculated_base_color = base_albedo_color.rgb * sampled_detail_color;
//      vec3 calculated_base_color = base_albedo_color.rgb * sampled_detail_color;

        // --- Ambient Occlusion Calculation ---
//      // --- Ambient Occlusion Calculation ---
        // Initializes a baseline ambient occlusion value representing no occlusion.
//      // Initializes a baseline ambient occlusion value representing no occlusion.
        float applied_ao_value = 1.0;
//      float applied_ao_value = 1.0;

        if (enable_hair_ambient_occlusion) {
//      if (enable_hair_ambient_occlusion) {
            // Samples the ambient occlusion texture using the pre-scaled UV varying variable.
//          // Samples the ambient occlusion texture using the pre-scaled UV varying variable.
            float sampled_hair_ao = texture(hair_ambient_occlusion_texture, scaled_ao_uv).r;
//          float sampled_hair_ao = texture(hair_ambient_occlusion_texture, scaled_ao_uv).r;

            // Interpolates between the un-occluded baseline (1.0) and the sampled value based on the intensity uniform.
//          // Interpolates between the un-occluded baseline (1.0) and the sampled value based on the intensity uniform.
            applied_ao_value = mix(1.0, sampled_hair_ao, hair_ambient_occlusion_intensity);
//          applied_ao_value = mix(1.0, sampled_hair_ao, hair_ambient_occlusion_intensity);

            // Assigns the result to Godot's built-in ambient occlusion property to affect standard indirect and ambient lighting.
//          // Assigns the result to Godot's built-in ambient occlusion property to affect standard indirect and ambient lighting.
            AO = applied_ao_value;
//          AO = applied_ao_value;

            // Ensures the ambient occlusion also affects direct light sources, enhancing the grounding of folds and deep geometry.
//          // Ensures the ambient occlusion also affects direct light sources, enhancing the grounding of folds and deep geometry.
            AO_LIGHT_AFFECT = 1.0;
//          AO_LIGHT_AFFECT = 1.0;
        }
//      }

        if (enable_film_grain_noise) {
//      if (enable_film_grain_noise) {
            // Calculates the pseudo-random grain noise using the fragment's screen UV coordinates.
//          // Calculates the pseudo-random grain noise using the fragment's screen UV coordinates.
            // This generates a high-frequency noise pattern to simulate film grain.
//          // This generates a high-frequency noise pattern to simulate film grain.
            float generated_grain_noise_value = compute_random_grain_noise(SCREEN_UV + vec2(TIME) * film_grain_noise_speed);
//          float generated_grain_noise_value = compute_random_grain_noise(SCREEN_UV + vec2(TIME) * film_grain_noise_speed);

            // Remaps the grain noise value from the range [0.0, 1.0] to [-1.0, 1.0] and scales it by the chosen intensity.
//          // Remaps the grain noise value from the range [0.0, 1.0] to [-1.0, 1.0] and scales it by the chosen intensity.
            float applied_grain_color_adjustment = (generated_grain_noise_value - 0.5) * film_grain_intensity;
//          float applied_grain_color_adjustment = (generated_grain_noise_value - 0.5) * film_grain_intensity;

            // Applies the computed grain adjustment uniformly across the RGB channels of the base color.
//          // Applies the computed grain adjustment uniformly across the RGB channels of the base color.
            calculated_base_color = clamp(calculated_base_color + vec3(applied_grain_color_adjustment), 0.0, 1.0);
//          calculated_base_color = clamp(calculated_base_color + vec3(applied_grain_color_adjustment), 0.0, 1.0);
        }
//      }

        // By passing this variable as a varying, the shader avoids redundant texture sampling in the per-pixel lighting pass.
//      // By passing this variable as a varying, the shader avoids redundant texture sampling in the per-pixel lighting pass.
        stored_raw_base_color = calculated_base_color * backlight_transmission_intensity;
//      stored_raw_base_color = calculated_base_color * backlight_transmission_intensity;

        // Uses luminance (not full RGB) to avoid double-coloring, since the LUT already encodes sigma_a absorption color.
//      // Uses luminance (not full RGB) to avoid double-coloring, since the LUT already encodes sigma_a absorption color.
        stored_detail_luminance = dot(calculated_base_color, constant_luminance_weights) * direct_light_intensity_scale * direct_bsdf_intensity;
//      stored_detail_luminance = dot(calculated_base_color, constant_luminance_weights) * direct_light_intensity_scale * direct_bsdf_intensity;

        // Defines the orientation of the hair strands for anisotropic highlights.
//      // Defines the orientation of the hair strands for anisotropic highlights.
        // Hair strands typically follow the V axis of the UV map (top to bottom).
//      // Hair strands typically follow the V axis of the UV map (top to bottom).
        // In Godot's tangent space, the V axis corresponds to the BINORMAL.
//      // In Godot's tangent space, the V axis corresponds to the BINORMAL.
        // If your specular highlights appear rotated by 90 degrees, change this assignment to TANGENT.
//      // If your specular highlights appear rotated by 90 degrees, change this assignment to TANGENT.
        vec3 normalized_hair_tangent = normalize(BINORMAL);
//      vec3 normalized_hair_tangent = normalize(BINORMAL);
        stored_hair_tangent = normalized_hair_tangent;
//      stored_hair_tangent = normalized_hair_tangent;

        // --- Precompute View-Dependent Math ---
//      // --- Precompute View-Dependent Math ---
        // Hoisting per-pixel operations out of the per-light function improves performance when multiple light sources illuminate the hair.
//      // Hoisting per-pixel operations out of the per-light function improves performance when multiple light sources illuminate the hair.
        float sine_angle_theta_outgoing = dot(VIEW, normalized_hair_tangent);
//      float sine_angle_theta_outgoing = dot(VIEW, normalized_hair_tangent);
        float angle_theta_outgoing = asin(clamp(sine_angle_theta_outgoing, -1.0, 1.0));
//      float angle_theta_outgoing = asin(clamp(sine_angle_theta_outgoing, -1.0, 1.0));
        float texture_coordinate_v = (angle_theta_outgoing * constant_inverse_pi) + 0.5;
//      float texture_coordinate_v = (angle_theta_outgoing * constant_inverse_pi) + 0.5;
        float cosine_squared_outgoing = max(1.0 - (sine_angle_theta_outgoing * sine_angle_theta_outgoing), 0.0);
//      float cosine_squared_outgoing = max(1.0 - (sine_angle_theta_outgoing * sine_angle_theta_outgoing), 0.0);

        // Packs data to pass to the per-light function, saving varying slots.
//      // Packs data to pass to the per-light function, saving varying slots.
        stored_outgoing_data = vec3(texture_coordinate_v, cosine_squared_outgoing, sine_angle_theta_outgoing);
//      stored_outgoing_data = vec3(texture_coordinate_v, cosine_squared_outgoing, sine_angle_theta_outgoing);

        if (enable_custom_ambient_lighting) {
//      if (enable_custom_ambient_lighting) {
            // Suppresses Godot's default Physically Based Rendering (PBR) ambient lighting by setting the ALBEDO to black.
//          // Suppresses Godot's default Physically Based Rendering (PBR) ambient lighting by setting the ALBEDO to black.
            // This prevents the built-in ambient light from being added to the output.
//          // This prevents the built-in ambient light from being added to the output.
            ALBEDO = vec3(0.0);
//          ALBEDO = vec3(0.0);

            // Adds the output of the custom ambient light function to the EMISSION channel.
//          // Adds the output of the custom ambient light function to the EMISSION channel.
            // This ensures the custom ambient light bypasses standard shading and is rendered as an unlit color.
//          // This ensures the custom ambient light bypasses standard shading and is rendered as an unlit color.
            vec3 computed_custom_ambient = compute_custom_ambient_lighting(calculated_base_color, NORMAL, normalized_hair_tangent, VIEW);
//          vec3 computed_custom_ambient = compute_custom_ambient_lighting(calculated_base_color, NORMAL, normalized_hair_tangent, VIEW);

            // Multiplies the custom ambient result by the ambient occlusion value to ensure consistent shadowing across lighting modes.
//          // Multiplies the custom ambient result by the ambient occlusion value to ensure consistent shadowing across lighting modes.
            EMISSION = computed_custom_ambient * applied_ao_value;
//          EMISSION = computed_custom_ambient * applied_ao_value;
        } else {
//      } else {
            // The albedo drives the base color in Godot's forward pass for ambient and indirect light calculations.
//          // The albedo drives the base color in Godot's forward pass for ambient and indirect light calculations.
            // It is scaled by the indirect ambient albedo scale uniform to ensure ambient light does not unnaturally wash out the hair texture.
//          // It is scaled by the indirect ambient albedo scale uniform to ensure ambient light does not unnaturally wash out the hair texture.
            ALBEDO = calculated_base_color * indirect_ambient_albedo_scale;
//          ALBEDO = calculated_base_color * indirect_ambient_albedo_scale;
        }
//      }

        // Ensures the shader does not output default PBR metallic or roughness values.
//      // Ensures the shader does not output default PBR metallic or roughness values.
        // Setting the ROUGHNESS to 1.0 and the METALLIC to 0.0 prevents Godot's standard reflections from interfering with the custom BSDF lighting function.
//      // Setting the ROUGHNESS to 1.0 and the METALLIC to 0.0 prevents Godot's standard reflections from interfering with the custom BSDF lighting function.
        ROUGHNESS = 1.0;
//      ROUGHNESS = 1.0;
        METALLIC = 0.0;
//      METALLIC = 0.0;

        // Samples the normal map texture using the pre-scaled UV varying variable.
//      // Samples the normal map texture using the pre-scaled UV varying variable.
        vec3 sampled_hair_normal_map_color = texture(hair_normal_map_texture, scaled_normal_uv).rgb;
//      vec3 sampled_hair_normal_map_color = texture(hair_normal_map_texture, scaled_normal_uv).rgb;

        // Applies the sampled normal map to the built-in PBR normal map property.
//      // Applies the sampled normal map to the built-in PBR normal map property.
        NORMAL_MAP = sampled_hair_normal_map_color;
//      NORMAL_MAP = sampled_hair_normal_map_color;

        // Scales the overall intensity of the normal map effect.
//      // Scales the overall intensity of the normal map effect.
        NORMAL_MAP_DEPTH = hair_normal_map_intensity;
//      NORMAL_MAP_DEPTH = hair_normal_map_intensity;
    }
//  }

    void light() {
//  void light() {
        // Retrieves the pre-normalized tangent from the varying variable.
//      // Retrieves the pre-normalized tangent from the varying variable.
        vec3 tangent_direction = stored_hair_tangent;
//      vec3 tangent_direction = stored_hair_tangent;
        vec3 light_direction = LIGHT;
//      vec3 light_direction = LIGHT;
        vec3 view_direction = VIEW;
//      vec3 view_direction = VIEW;

        // --- Retrieve Precomputed View-Dependent Math ---
//      // --- Retrieve Precomputed View-Dependent Math ---
        float texture_coordinate_v = stored_outgoing_data.x;
//      float texture_coordinate_v = stored_outgoing_data.x;
        float cosine_squared_outgoing = stored_outgoing_data.y;
//      float cosine_squared_outgoing = stored_outgoing_data.y;
        float sine_angle_theta_outgoing = stored_outgoing_data.z;
//      float sine_angle_theta_outgoing = stored_outgoing_data.z;

        // Step 1: Calculates the longitudinal inclinations (theta angles) for the incident light.
//      // Step 1: Calculates the longitudinal inclinations (theta angles) for the incident light.
        float sine_angle_theta_incident = dot(light_direction, tangent_direction);
//      float sine_angle_theta_incident = dot(light_direction, tangent_direction);
        float angle_theta_incident = asin(clamp(sine_angle_theta_incident, -1.0, 1.0));
//      float angle_theta_incident = asin(clamp(sine_angle_theta_incident, -1.0, 1.0));

        // Maps the incident longitudinal angle to the U texture coordinate.
//      // Maps the incident longitudinal angle to the U texture coordinate.
        float texture_coordinate_u = (angle_theta_incident * constant_inverse_pi) + 0.5;
//      float texture_coordinate_u = (angle_theta_incident * constant_inverse_pi) + 0.5;

        // Step 2: Calculates the azimuthal difference (phi angle).
//      // Step 2: Calculates the azimuthal difference (phi angle).
        float cosine_squared_incident = max(1.0 - (sine_angle_theta_incident * sine_angle_theta_incident), 0.0);
//      float cosine_squared_incident = max(1.0 - (sine_angle_theta_incident * sine_angle_theta_incident), 0.0);

        // Precomputes the dot product of the view and light vectors, reused for azimuthal projection and backlight scattering.
//      // Precomputes the dot product of the view and light vectors, reused for azimuthal projection and backlight scattering.
        float dot_view_light = dot(view_direction, light_direction);
//      float dot_view_light = dot(view_direction, light_direction);

        // Calculates the product of the squared cosines.
//      // Calculates the product of the squared cosines.
        float squared_lengths_product = cosine_squared_incident * cosine_squared_outgoing;
//      float squared_lengths_product = cosine_squared_incident * cosine_squared_outgoing;

        // Creates a mask using the step function to replace the conditional branch.
//      // Creates a mask using the step function to replace the conditional branch.
        // It returns 1.0 only if both the incident and outgoing squared cosines are above the threshold.
//      // It returns 1.0 only if both the incident and outgoing squared cosines are above the threshold.
        float valid_product_mask = step(0.000001, cosine_squared_incident) * step(0.000001, cosine_squared_outgoing);
//      float valid_product_mask = step(0.000001, cosine_squared_incident) * step(0.000001, cosine_squared_outgoing);

        // Calculates the dot product of the projected light and view vectors.
//      // Calculates the dot product of the projected light and view vectors.
        float dot_product_projected_light_and_view = dot_view_light - (sine_angle_theta_incident * sine_angle_theta_outgoing);
//      float dot_product_projected_light_and_view = dot_view_light - (sine_angle_theta_incident * sine_angle_theta_outgoing);

        // Safely calculates the inverse square root by ensuring the input is strictly positive using the max() function.
//      // Safely calculates the inverse square root by ensuring the input is strictly positive using the max() function.
        float inverse_projected_lengths_product = inversesqrt(max(squared_lengths_product, 0.000000000001));
//      float inverse_projected_lengths_product = inversesqrt(max(squared_lengths_product, 0.000000000001));

        // Calculates the azimuthal difference angle (phi).
//      // Calculates the azimuthal difference angle (phi).
        float calculated_angle_phi_difference = acos(clamp(dot_product_projected_light_and_view * inverse_projected_lengths_product, -1.0, 1.0));
//      float calculated_angle_phi_difference = acos(clamp(dot_product_projected_light_and_view * inverse_projected_lengths_product, -1.0, 1.0));

        float angle_phi_difference = calculated_angle_phi_difference;
//      float angle_phi_difference = calculated_angle_phi_difference;

        // Maps the azimuthal difference from the range [0, PI] to the normalized texture coordinate [0.0, 1.0] for the W axis (depth of the 3D texture).
//      // Maps the azimuthal difference from the range [0, PI] to the normalized texture coordinate [0.0, 1.0] for the W axis (depth of the 3D texture).
        float texture_coordinate_w = angle_phi_difference * constant_inverse_pi;
//      float texture_coordinate_w = angle_phi_difference * constant_inverse_pi;

        // Applies a half-texel correction specific to the 64x64x64 LUT dimensions.
//      // Applies a half-texel correction specific to the 64x64x64 LUT dimensions.
        // This prevents sampling artifacts and interpolation errors near the edges of the 3D texture volume.
//      // This prevents sampling artifacts and interpolation errors near the edges of the 3D texture volume.
        vec3 texture_coordinates_3d = vec3(texture_coordinate_u, texture_coordinate_v, texture_coordinate_w);
//      vec3 texture_coordinates_3d = vec3(texture_coordinate_u, texture_coordinate_v, texture_coordinate_w);
        texture_coordinates_3d = (texture_coordinates_3d * constant_lut_half_texel_scale) + constant_lut_half_texel_offset;
//      texture_coordinates_3d = (texture_coordinates_3d * constant_lut_half_texel_scale) + constant_lut_half_texel_offset;

        // Step 3: Samples the pre-baked BSDF Look-Up Table.
//      // Step 3: Samples the pre-baked BSDF Look-Up Table.
        // Evaluates the complex hair scattering function by sampling the 3D texture at the calculated (U, V, W) coordinates.
//      // Evaluates the complex hair scattering function by sampling the 3D texture at the calculated (U, V, W) coordinates.
        // Applies valid_product_mask to fade out the LUT when azimuthal angle is degenerate (light/view tangent-aligned).
//      // Applies valid_product_mask to fade out the LUT when azimuthal angle is degenerate (light/view tangent-aligned).
        vec3 sampled_brdf_evaluation = texture(hair_bsdf_texture_lut, texture_coordinates_3d).rgb * valid_product_mask;
//      vec3 sampled_brdf_evaluation = texture(hair_bsdf_texture_lut, texture_coordinates_3d).rgb * valid_product_mask;

        // Step 4: Computes the final output lighting.
//      // Step 4: Computes the final output lighting.
        // Combines the sampled BSDF with the base color, light color, attenuation, and custom transmission properties.
//      // Combines the sampled BSDF with the base color, light color, attenuation, and custom transmission properties.

        // Direct BSDF Evaluation: Multiplies the BSDF sample by the light color, attenuation (shadows/distance), and baked luminance/specular intensity.
//      // Direct BSDF Evaluation: Multiplies the BSDF sample by the light color, attenuation (shadows/distance), and baked luminance/specular intensity.
        vec3 evaluated_direct_light = sampled_brdf_evaluation * LIGHT_COLOR * ATTENUATION * stored_detail_luminance;
//      vec3 evaluated_direct_light = sampled_brdf_evaluation * LIGHT_COLOR * ATTENUATION * stored_detail_luminance;

        // Fake Light Transmission (Subsurface Scattering): Simulates light bleeding through the hair volume.
//      // Fake Light Transmission (Subsurface Scattering): Simulates light bleeding through the hair volume.
        // Applies a wrap lighting model to soften the terminator and allow light to wrap around the geometry.
//      // Applies a wrap lighting model to soften the terminator and allow light to wrap around the geometry.
        float constant_transmission_wrap_factor = 0.5;
//      float constant_transmission_wrap_factor = 0.5;
        float calculated_wrap_transmission_dot = max(dot(NORMAL, -light_direction) + constant_transmission_wrap_factor, 0.0) / (1.0 + constant_transmission_wrap_factor);
//      float calculated_wrap_transmission_dot = max(dot(NORMAL, -light_direction) + constant_transmission_wrap_factor, 0.0) / (1.0 + constant_transmission_wrap_factor);
        float computed_backlight_scatter = (calculated_wrap_transmission_dot * calculated_wrap_transmission_dot) * max(-dot_view_light, 0.0);
//      float computed_backlight_scatter = (calculated_wrap_transmission_dot * calculated_wrap_transmission_dot) * max(-dot_view_light, 0.0);
        vec3 evaluated_transmission = stored_raw_base_color * LIGHT_COLOR * computed_backlight_scatter;
//      vec3 evaluated_transmission = stored_raw_base_color * LIGHT_COLOR * computed_backlight_scatter;

        // Analytical Anisotropic Specular Highlight (Kajiya-Kay): Adds crisp strand reflections over the soft LUT base.
//      // Analytical Anisotropic Specular Highlight (Kajiya-Kay): Adds crisp strand reflections over the soft LUT base.
        float calculated_specular_term = pow(max(sqrt(cosine_squared_incident * cosine_squared_outgoing) + sine_angle_theta_incident * sine_angle_theta_outgoing, 0.0), hair_specular_power);
//      float calculated_specular_term = pow(max(sqrt(cosine_squared_incident * cosine_squared_outgoing) + sine_angle_theta_incident * sine_angle_theta_outgoing, 0.0), hair_specular_power);
        vec3 evaluated_analytical_specular = LIGHT_COLOR * ATTENUATION * calculated_specular_term * direct_bsdf_intensity;
//      vec3 evaluated_analytical_specular = LIGHT_COLOR * ATTENUATION * calculated_specular_term * direct_bsdf_intensity;

        // Multiple-Scattering Approximation: Simulates light bouncing between strands, crucial for light-colored hair.
//      // Multiple-Scattering Approximation: Simulates light bouncing between strands, crucial for light-colored hair.
        float calculated_multiple_scatter_factor = smoothstep(-0.5, 1.0, dot(NORMAL, light_direction));
//      float calculated_multiple_scatter_factor = smoothstep(-0.5, 1.0, dot(NORMAL, light_direction));
        vec3 evaluated_multiple_scatter = stored_raw_base_color * stored_raw_base_color * calculated_multiple_scatter_factor * LIGHT_COLOR * ATTENUATION * 0.5;
//      vec3 evaluated_multiple_scatter = stored_raw_base_color * stored_raw_base_color * calculated_multiple_scatter_factor * LIGHT_COLOR * ATTENUATION * 0.5;

        // View-Dependent Fresnel Rim Lighting: Illuminates the silhouette to prevent a flat, solid appearance.
//      // View-Dependent Fresnel Rim Lighting: Illuminates the silhouette to prevent a flat, solid appearance.
        float calculated_fresnel_rim_factor = pow(1.0 - clamp(dot(NORMAL, view_direction), 0.0, 1.0), 3.0);
//      float calculated_fresnel_rim_factor = pow(1.0 - clamp(dot(NORMAL, view_direction), 0.0, 1.0), 3.0);
        float calculated_rim_light_intensity = smoothstep(0.0, 1.0, dot(NORMAL, light_direction)) * calculated_fresnel_rim_factor;
//      float calculated_rim_light_intensity = smoothstep(0.0, 1.0, dot(NORMAL, light_direction)) * calculated_fresnel_rim_factor;
        vec3 evaluated_rim_light = stored_raw_base_color * LIGHT_COLOR * ATTENUATION * calculated_rim_light_intensity * 0.5;
//      vec3 evaluated_rim_light = stored_raw_base_color * LIGHT_COLOR * ATTENUATION * calculated_rim_light_intensity * 0.5;

        // Combines the evaluated direct light, transmission, specular, multiple scattering, and rim lighting.
//      // Combines the evaluated direct light, transmission, specular, multiple scattering, and rim lighting.
        // Blends in a small baseline amount of transmission even in shadowed areas (where ATTENUATION approaches 0.0) to simulate global multiple scattering.
//      // Blends in a small baseline amount of transmission even in shadowed areas (where ATTENUATION approaches 0.0) to simulate global multiple scattering.
        vec3 combined_final_light = evaluated_direct_light + evaluated_analytical_specular + evaluated_multiple_scatter + evaluated_rim_light + (evaluated_transmission * mix(0.1, 1.0, ATTENUATION));
//      vec3 combined_final_light = evaluated_direct_light + evaluated_analytical_specular + evaluated_multiple_scatter + evaluated_rim_light + (evaluated_transmission * mix(0.1, 1.0, ATTENUATION));

        DIFFUSE_LIGHT += combined_final_light;
//      DIFFUSE_LIGHT += combined_final_light;
    }
//  }
