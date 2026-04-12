    shader_type spatial;
//  shader_type spatial;
    render_mode blend_mix, depth_draw_opaque, cull_disabled, diffuse_burley, specular_schlick_ggx, depth_prepass_alpha;
//  render_mode blend_mix, depth_draw_opaque, cull_disabled, diffuse_burley, specular_schlick_ggx, depth_prepass_alpha;

    // The 3D texture containing the baked PBRT hair Bidirectional Scattering Distribution Function (BSDF)
//  // The 3D texture containing the baked PBRT hair Bidirectional Scattering Distribution Function (BSDF)
    // This Look-Up Table (LUT) stores the precomputed complex hair scattering properties.
//  // This Look-Up Table (LUT) stores the precomputed complex hair scattering properties.
    // This should be imported in Godot as a 3D Texture with dimensions 64x64x64 (8 columns, 8 rows).
//  // This should be imported in Godot as a 3D Texture with dimensions 64x64x64 (8 columns, 8 rows).
    uniform sampler3D hair_bsdf_texture_lut : hint_default_black, filter_linear, repeat_disable;
//  uniform sampler3D hair_bsdf_texture_lut : hint_default_black, filter_linear, repeat_disable;

    // Basic Hair Material Properties
//  // Basic Hair Material Properties
    // Defines the foundational visual characteristics of the hair surface.
//  // Defines the foundational visual characteristics of the hair surface.
    uniform vec3 base_albedo_color : source_color = vec3(1.0, 1.0, 1.0);
//  uniform vec3 base_albedo_color : source_color = vec3(1.0, 1.0, 1.0);
    uniform sampler2D detail_albedo_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D detail_albedo_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
    uniform vec2 detail_texture_uv_scale = vec2(1.0, 1.0);
//  uniform vec2 detail_texture_uv_scale = vec2(1.0, 1.0);
    uniform float direct_specular_intensity : hint_range(0.0, 5.0) = 0.3;
//  uniform float direct_specular_intensity : hint_range(0.0, 5.0) = 0.3;

    // Advanced Lighting and Shading Properties
//  // Advanced Lighting and Shading Properties
    // Controls how the hair reacts to both direct light sources and indirect ambient lighting.
//  // Controls how the hair reacts to both direct light sources and indirect ambient lighting.
    uniform float indirect_ambient_albedo_scale : hint_range(0.0, 1.0) = 1.0; // Increased default for brighter ambient
//  uniform float indirect_ambient_albedo_scale : hint_range(0.0, 1.0) = 1.0; // Increased default for brighter ambient
    uniform float direct_light_albedo_scale : hint_range(1.0, 100.0) = 100.0;
//  uniform float direct_light_albedo_scale : hint_range(1.0, 100.0) = 100.0;
    uniform float backlight_transmission_intensity : hint_range(0.0, 2.0) = 2.0; // Controls light passing through hair
//  uniform float backlight_transmission_intensity : hint_range(0.0, 2.0) = 2.0; // Controls light passing through hair
    uniform bool enable_custom_ambient_lighting = false;
//  uniform bool enable_custom_ambient_lighting = false;

    // Post-Processing and Stylization Properties
//  // Post-Processing and Stylization Properties
    // Toggles the film grain noise overlay.
//  // Toggles the film grain noise overlay.
    uniform bool enable_film_grain_noise = false;
//  uniform bool enable_film_grain_noise = false;
    // Defines stylistic overlays such as film grain noise.
//  // Defines stylistic overlays such as film grain noise.
    uniform float film_grain_intensity : hint_range(0.0, 1.0) = 0.01;
//  uniform float film_grain_intensity : hint_range(0.0, 1.0) = 0.01;

    varying vec3 surface_hair_tangent;
//  varying vec3 surface_hair_tangent;
    varying vec3 stored_raw_base_color;
//  varying vec3 stored_raw_base_color;

    // Custom function for calculating ambient/indirect lighting.
//  // Custom function for calculating ambient/indirect lighting.
    // This allows developers to inject specific logic for global illumination or stylistic ambient shading.
//  // This allows developers to inject specific logic for global illumination or stylistic ambient shading.
    vec3 compute_custom_ambient_lighting(vec3 surface_base_color_rgb, vec3 surface_normal, vec3 surface_tangent, vec3 view_direction) {
//  vec3 compute_custom_ambient_lighting(vec3 surface_base_color_rgb, vec3 surface_normal, vec3 surface_tangent, vec3 view_direction) {
        // Add your custom ambient lighting logic here to override Godot's default environment.
//      // Add your custom ambient lighting logic here to override Godot's default environment.
        // For example, you could implement a fake directional hemispheric light, a stylized constant color, or a custom cubemap sample.
//      // For example, you could implement a fake directional hemispheric light, a stylized constant color, or a custom cubemap sample.
        vec3 ambient_color = vec3(0.05); // Example placeholder
//      vec3 ambient_color = vec3(0.05); // Example placeholder
        return surface_base_color_rgb * ambient_color;
//      return surface_base_color_rgb * ambient_color;
    }
//  }

    // Generates a pseudo-random value based on a 2D input coordinate.
//  // Generates a pseudo-random value based on a 2D input coordinate.
    // This function is utilized to synthesize a static noise pattern for the grainy visual effect.
//  // This function is utilized to synthesize a static noise pattern for the grainy visual effect.
    float compute_random_grain_noise(vec2 input_screen_coordinate) {
//  float compute_random_grain_noise(vec2 input_screen_coordinate) {
        // Compute a pseudo-random float utilizing a dot product and sine function hash.
//      // Compute a pseudo-random float utilizing a dot product and sine function hash.
        return fract(sin(dot(input_screen_coordinate, vec2(12.9898, 78.233))) * 43758.5453);
//      return fract(sin(dot(input_screen_coordinate, vec2(12.9898, 78.233))) * 43758.5453);
    }
//  }

    void vertex() {
//  void vertex() {
        // Standard vertex transform pass.
//      // Standard vertex transform pass.
        // This is left empty to allow Godot's default vertex processing to handle geometry transformations.
//      // This is left empty to allow Godot's default vertex processing to handle geometry transformations.
    }
//  }

    void fragment() {
//  void fragment() {
        vec4 sampled_detail_texture = texture(detail_albedo_texture, UV * detail_texture_uv_scale);
//      vec4 sampled_detail_texture = texture(detail_albedo_texture, UV * detail_texture_uv_scale);
        vec3 calculated_base_color = base_albedo_color * sampled_detail_texture.rgb;
//      vec3 calculated_base_color = base_albedo_color * sampled_detail_texture.rgb;

        if (enable_film_grain_noise) {
//      if (enable_film_grain_noise) {
            // Calculate the pseudo-random grain noise using the fragment's screen UV coordinates.
//          // Calculate the pseudo-random grain noise using the fragment's screen UV coordinates.
            // This generates a high-frequency noise pattern to simulate film grain.
//          // This generates a high-frequency noise pattern to simulate film grain.
            float generated_grain_noise_value = compute_random_grain_noise(SCREEN_UV);
//          float generated_grain_noise_value = compute_random_grain_noise(SCREEN_UV);

            // Remap the grain noise value from the range [0.0, 1.0] to [-1.0, 1.0] and scale by the chosen intensity.
//          // Remap the grain noise value from the range [0.0, 1.0] to [-1.0, 1.0] and scale by the chosen intensity.
            float applied_grain_color_adjustment = (generated_grain_noise_value - 0.5) * film_grain_intensity;
//          float applied_grain_color_adjustment = (generated_grain_noise_value - 0.5) * film_grain_intensity;

            // Apply the computed grain adjustment uniformly across the RGB channels of the base color.
//          // Apply the computed grain adjustment uniformly across the RGB channels of the base color.
            calculated_base_color = clamp(calculated_base_color + vec3(applied_grain_color_adjustment), 0.0, 1.0);
//          calculated_base_color = clamp(calculated_base_color + vec3(applied_grain_color_adjustment), 0.0, 1.0);
        }
//      }

        // Store the raw base color for the light() function to optimize performance.
//      // Store the raw base color for the light() function to optimize performance.
        // By passing this as a varying, we avoid redundant texture sampling in the per-pixel lighting pass.
//      // By passing this as a varying, we avoid redundant texture sampling in the per-pixel lighting pass.
        stored_raw_base_color = calculated_base_color;
//      stored_raw_base_color = calculated_base_color;

        // Apply alpha from the detail texture to determine hair transparency and sorting.
//      // Apply alpha from the detail texture to determine hair transparency and sorting.
        ALPHA = sampled_detail_texture.a;
//      ALPHA = sampled_detail_texture.a;

        // Define the orientation of the hair strands for anisotropic highlights.
//      // Define the orientation of the hair strands for anisotropic highlights.
        // Hair strands typically follow the V axis of the UV map (top to bottom).
//      // Hair strands typically follow the V axis of the UV map (top to bottom).
        // In Godot's tangent space, the V axis corresponds to the BINORMAL.
//      // In Godot's tangent space, the V axis corresponds to the BINORMAL.
        // If your specular highlights appear rotated by 90 degrees, change this assignment to TANGENT.
//      // If your specular highlights appear rotated by 90 degrees, change this assignment to TANGENT.
        surface_hair_tangent = BINORMAL;
//      surface_hair_tangent = BINORMAL;

        if (enable_custom_ambient_lighting) {
//      if (enable_custom_ambient_lighting) {
            // Suppress Godot's default Physically Based Rendering (PBR) ambient lighting
//          // Suppress Godot's default Physically Based Rendering (PBR) ambient lighting
            // by setting ALBEDO to black. This prevents the built-in ambient light from being added.
//          // by setting ALBEDO to black. This prevents the built-in ambient light from being added.
            ALBEDO = vec3(0.0);
//          ALBEDO = vec3(0.0);

            // Add the output of the custom ambient light function to the EMISSION channel.
//          // Add the output of the custom ambient light function to the EMISSION channel.
            // This ensures the custom ambient light bypasses standard shading and is rendered as unlit color.
//          // This ensures the custom ambient light bypasses standard shading and is rendered as unlit color.
            vec3 normalized_hair_tangent = normalize(surface_hair_tangent);
//          vec3 normalized_hair_tangent = normalize(surface_hair_tangent);
            vec3 normalized_view_direction = normalize(VIEW);
//          vec3 normalized_view_direction = normalize(VIEW);
            vec3 computed_custom_ambient = compute_custom_ambient_lighting(calculated_base_color, NORMAL, normalized_hair_tangent, normalized_view_direction);
//          vec3 computed_custom_ambient = compute_custom_ambient_lighting(calculated_base_color, NORMAL, normalized_hair_tangent, normalized_view_direction);
            EMISSION = computed_custom_ambient;
//          EMISSION = computed_custom_ambient;
        } else {
//      } else {
            // Albedo drives the base color in Godot's forward pass for ambient and indirect light calculation.
//          // Albedo drives the base color in Godot's forward pass for ambient and indirect light calculation.
            // We scale it by the indirect_ambient_albedo_scale uniform to ensure ambient light doesn't unnaturally wash out the hair texture.
//          // We scale it by the indirect_ambient_albedo_scale uniform to ensure ambient light doesn't unnaturally wash out the hair texture.
            ALBEDO = calculated_base_color * indirect_ambient_albedo_scale;
//          ALBEDO = calculated_base_color * indirect_ambient_albedo_scale;
        }
//      }

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
        // Establish the local coordinate frame for the hair strand.
//      // Establish the local coordinate frame for the hair strand.
        // We assume the tangent vector (surface_hair_tangent) points longitudinally ALONG the hair strand.
//      // We assume the tangent vector (surface_hair_tangent) points longitudinally ALONG the hair strand.
        vec3 tangent_direction = normalize(surface_hair_tangent);
//      vec3 tangent_direction = normalize(surface_hair_tangent);
        vec3 light_direction = normalize(LIGHT);
//      vec3 light_direction = normalize(LIGHT);
        vec3 view_direction = normalize(VIEW);
//      vec3 view_direction = normalize(VIEW);

        // 1. Calculate Longitudinal Inclinations (Theta angles)
//      // 1. Calculate Longitudinal Inclinations (Theta angles)
        // These angles represent the elevation of the incident light and view vectors relative to the perpendicular plane of the hair strand.
//      // These angles represent the elevation of the incident light and view vectors relative to the perpendicular plane of the hair strand.
        float sine_angle_theta_incident = dot(light_direction, tangent_direction);
//      float sine_angle_theta_incident = dot(light_direction, tangent_direction);
        float sine_angle_theta_outgoing = dot(view_direction, tangent_direction);
//      float sine_angle_theta_outgoing = dot(view_direction, tangent_direction);

        float angle_theta_incident = asin(clamp(sine_angle_theta_incident, -1.0, 1.0));
//      float angle_theta_incident = asin(clamp(sine_angle_theta_incident, -1.0, 1.0));
        float angle_theta_outgoing = asin(clamp(sine_angle_theta_outgoing, -1.0, 1.0));
//      float angle_theta_outgoing = asin(clamp(sine_angle_theta_outgoing, -1.0, 1.0));

        // Map the longitudinal angles from the range [-PI/2, PI/2] to the normalized texture coordinates [0.0, 1.0] for U and V axes.
//      // Map the longitudinal angles from the range [-PI/2, PI/2] to the normalized texture coordinates [0.0, 1.0] for U and V axes.
        float texture_coordinate_u = (angle_theta_incident + (PI / 2.0)) / PI;
//      float texture_coordinate_u = (angle_theta_incident + (PI / 2.0)) / PI;
        float texture_coordinate_v = (angle_theta_outgoing + (PI / 2.0)) / PI;
//      float texture_coordinate_v = (angle_theta_outgoing + (PI / 2.0)) / PI;

        // 2. Calculate Azimuthal Difference (Phi angle)
//      // 2. Calculate Azimuthal Difference (Phi angle)
        // This is the relative angle between the light and view vectors when projected onto the normal plane (perpendicular to the hair strand).
//      // This is the relative angle between the light and view vectors when projected onto the normal plane (perpendicular to the hair strand).
        vec3 light_projected_on_normal_plane = light_direction - tangent_direction * sine_angle_theta_incident;
//      vec3 light_projected_on_normal_plane = light_direction - tangent_direction * sine_angle_theta_incident;
        vec3 view_projected_on_normal_plane = view_direction - tangent_direction * sine_angle_theta_outgoing;
//      vec3 view_projected_on_normal_plane = view_direction - tangent_direction * sine_angle_theta_outgoing;

        float length_light_projected = length(light_projected_on_normal_plane);
//      float length_light_projected = length(light_projected_on_normal_plane);
        float length_view_projected = length(view_projected_on_normal_plane);
//      float length_view_projected = length(view_projected_on_normal_plane);

        float angle_phi_difference = 0.0;
//      float angle_phi_difference = 0.0;
        if (length_light_projected > 0.001 && length_view_projected > 0.001) {
//      if (length_light_projected > 0.001 && length_view_projected > 0.001) {
            angle_phi_difference = acos(clamp(dot(light_projected_on_normal_plane / length_light_projected, view_projected_on_normal_plane / length_view_projected), -1.0, 1.0));
//          angle_phi_difference = acos(clamp(dot(light_projected_on_normal_plane / length_light_projected, view_projected_on_normal_plane / length_view_projected), -1.0, 1.0));
        }
//      }

        // Map the azimuthal difference from the range [0, PI] to the normalized texture coordinate [0.0, 1.0] for the W axis (depth of the 3D texture).
//      // Map the azimuthal difference from the range [0, PI] to the normalized texture coordinate [0.0, 1.0] for the W axis (depth of the 3D texture).
        float texture_coordinate_w = angle_phi_difference / PI;
//      float texture_coordinate_w = angle_phi_difference / PI;

        // Apply a half-texel correction specific to our 64x64x64 LUT dimensions.
//      // Apply a half-texel correction specific to our 64x64x64 LUT dimensions.
        // This prevents sampling artifacts and interpolation errors near the edges of the 3D texture volume.
//      // This prevents sampling artifacts and interpolation errors near the edges of the 3D texture volume.
        float correction_texel_size = 1.0 / 64.0;
//      float correction_texel_size = 1.0 / 64.0;
        vec3 texture_coordinates_3d = vec3(texture_coordinate_u, texture_coordinate_v, texture_coordinate_w);
//      vec3 texture_coordinates_3d = vec3(texture_coordinate_u, texture_coordinate_v, texture_coordinate_w);
        texture_coordinates_3d = texture_coordinates_3d * (1.0 - correction_texel_size) + (correction_texel_size * 0.5);
//      texture_coordinates_3d = texture_coordinates_3d * (1.0 - correction_texel_size) + (correction_texel_size * 0.5);

        // 3. Sample the Pre-baked BSDF Look-Up Table
//      // 3. Sample the Pre-baked BSDF Look-Up Table
        // We evaluate the complex hair scattering function by sampling the 3D texture at the calculated (U, V, W) coordinates.
//      // We evaluate the complex hair scattering function by sampling the 3D texture at the calculated (U, V, W) coordinates.
        vec3 sampled_brdf_evaluation = texture(hair_bsdf_texture_lut, texture_coordinates_3d).rgb;
//      vec3 sampled_brdf_evaluation = texture(hair_bsdf_texture_lut, texture_coordinates_3d).rgb;

        // 4. Compute Final Output Lighting
//      // 4. Compute Final Output Lighting
        // We combine the sampled BSDF with base color, light color, attenuation, and custom transmission properties.
//      // We combine the sampled BSDF with base color, light color, attenuation, and custom transmission properties.
        vec3 scaled_base_color = stored_raw_base_color * direct_light_albedo_scale;
//      vec3 scaled_base_color = stored_raw_base_color * direct_light_albedo_scale;
        float calculated_detail_luminance = dot(scaled_base_color, vec3(0.299, 0.587, 0.114));
//      float calculated_detail_luminance = dot(scaled_base_color, vec3(0.299, 0.587, 0.114));

        // Direct BSDF Evaluation: Multiply the BSDF sample by light color, attenuation (shadows/distance), luminance, and specular intensity.
//      // Direct BSDF Evaluation: Multiply the BSDF sample by light color, attenuation (shadows/distance), luminance, and specular intensity.
        vec3 evaluated_direct_light = sampled_brdf_evaluation * LIGHT_COLOR * ATTENUATION * calculated_detail_luminance * direct_specular_intensity;
//      vec3 evaluated_direct_light = sampled_brdf_evaluation * LIGHT_COLOR * ATTENUATION * calculated_detail_luminance * direct_specular_intensity;

        // Fake Light Transmission (Subsurface Scattering): Simulate light bleeding through the hair volume.
//      // Fake Light Transmission (Subsurface Scattering): Simulate light bleeding through the hair volume.
        // This effect illuminates the back side of the hair mesh and softens harsh geometric shadows.
//      // This effect illuminates the back side of the hair mesh and softens harsh geometric shadows.
        float computed_backlight_scatter = pow(clamp(dot(view_direction, -light_direction), 0.0, 1.0), 2.0);
//      float computed_backlight_scatter = pow(clamp(dot(view_direction, -light_direction), 0.0, 1.0), 2.0);
        vec3 evaluated_transmission = stored_raw_base_color * LIGHT_COLOR * computed_backlight_scatter * backlight_transmission_intensity;
//      vec3 evaluated_transmission = stored_raw_base_color * LIGHT_COLOR * computed_backlight_scatter * backlight_transmission_intensity;

        // Combine the evaluated direct light and transmission.
//      // Combine the evaluated direct light and transmission.
        // We blend in a small baseline amount of transmission even in shadowed areas (where ATTENUATION approaches 0.0) to fake global multiple scattering.
//      // We blend in a small baseline amount of transmission even in shadowed areas (where ATTENUATION approaches 0.0) to fake global multiple scattering.
        vec3 combined_final_light = evaluated_direct_light + (evaluated_transmission * mix(0.1, 1.0, ATTENUATION));
//      vec3 combined_final_light = evaluated_direct_light + (evaluated_transmission * mix(0.1, 1.0, ATTENUATION));

        DIFFUSE_LIGHT += combined_final_light;
//      DIFFUSE_LIGHT += combined_final_light;
    }
//  }
