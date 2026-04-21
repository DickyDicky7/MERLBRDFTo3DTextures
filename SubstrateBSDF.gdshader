    shader_type spatial;
//  shader_type spatial;
    render_mode blend_mix, depth_draw_opaque, cull_disabled, diffuse_burley, specular_disabled;
//  render_mode blend_mix, depth_draw_opaque, cull_disabled, diffuse_burley, specular_disabled;

    // --- Basic Layered Material Properties ---
//  // --- Basic Layered Material Properties ---
    // The foundational color of the underlying substrate layer.
//  // The foundational color of the underlying substrate layer.
    uniform vec3 substrate_base_color : source_color = vec3(0.1, 0.5, 0.8);
//  uniform vec3 substrate_base_color : source_color = vec3(0.1, 0.5, 0.8);
    // Detail texture to provide color variation across the substrate surface.
//  // Detail texture to provide color variation across the substrate surface.
    uniform sampler2D substrate_base_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D substrate_base_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
    // Scale factor to tile the base detail texture across the mesh surface.
//  // Scale factor to tile the base detail texture across the mesh surface.
    uniform vec2 substrate_base_texture_uv_scale = vec2(1.0, 1.0);
//  uniform vec2 substrate_base_texture_uv_scale = vec2(1.0, 1.0);

    // The specular color representing the reflectance at normal incidence (F0) for the clearcoat layer.
//  // The specular color representing the reflectance at normal incidence (F0) for the clearcoat layer.
    // The default value is set to 4% reflectance, which is physically accurate for most common dielectric materials.
//  // The default value is set to 4% reflectance, which is physically accurate for most common dielectric materials.
    uniform vec3 clearcoat_specular_color : source_color = vec3(0.04);
//  uniform vec3 clearcoat_specular_color : source_color = vec3(0.04);
    // Texture to spatially vary the specular reflection intensity across the clearcoat layer.
//  // Texture to spatially vary the specular reflection intensity across the clearcoat layer.
    uniform sampler2D clearcoat_specular_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D clearcoat_specular_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
    // Scale factor to tile the specular texture across the mesh surface.
//  // Scale factor to tile the specular texture across the mesh surface.
    uniform vec2 clearcoat_specular_texture_uv_scale = vec2(1.0, 1.0);
//  uniform vec2 clearcoat_specular_texture_uv_scale = vec2(1.0, 1.0);

    // The perceived microscopic roughness of the clearcoat layer, which blurs the specular highlights.
//  // The perceived microscopic roughness of the clearcoat layer, which blurs the specular highlights.
    uniform float clearcoat_roughness : hint_range(0.001, 1.0) = 0.1;
//  uniform float clearcoat_roughness : hint_range(0.001, 1.0) = 0.1;
    // Texture to spatially vary the perceived microscopic roughness across the clearcoat layer.
//  // Texture to spatially vary the perceived microscopic roughness across the clearcoat layer.
    uniform sampler2D clearcoat_roughness_texture : hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D clearcoat_roughness_texture : hint_default_white, filter_linear_mipmap, repeat_enable;
    // Scale factor to tile the roughness texture across the mesh surface.
//  // Scale factor to tile the roughness texture across the mesh surface.
    uniform vec2 clearcoat_roughness_texture_uv_scale = vec2(1.0, 1.0);
//  uniform vec2 clearcoat_roughness_texture_uv_scale = vec2(1.0, 1.0);

    // --- Surface Detail Properties ---
//  // --- Surface Detail Properties ---
    // Normal map texture to add high-frequency micro-surface geometric details without increasing polygon count.
//  // Normal map texture to add high-frequency micro-surface geometric details without increasing polygon count.
    uniform sampler2D substrate_normal_map_texture : hint_normal, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D substrate_normal_map_texture : hint_normal, filter_linear_mipmap, repeat_enable;
    // Scale factor to tile the normal map texture across the mesh surface.
//  // Scale factor to tile the normal map texture across the mesh surface.
    uniform vec2 substrate_normal_map_texture_uv_scale = vec2(1.0, 1.0);
//  uniform vec2 substrate_normal_map_texture_uv_scale = vec2(1.0, 1.0);
    // Overall strength of the normal map geometric displacement effect.
//  // Overall strength of the normal map geometric displacement effect.
    uniform float substrate_normal_map_intensity : hint_range(0.0, 5.0) = 1.0;
//  uniform float substrate_normal_map_intensity : hint_range(0.0, 5.0) = 1.0;

    // --- Post-Processing Properties ---
//  // --- Post-Processing Properties ---
    // A multiplier for the overall intensity of calculated direct lighting, allowing for artistic adjustments.
//  // A multiplier for the overall intensity of calculated direct lighting, allowing for artistic adjustments.
    uniform float direct_light_intensity : hint_range(0.0, 5.0) = 1.0;
//  uniform float direct_light_intensity : hint_range(0.0, 5.0) = 1.0;

    // Varying variable to securely pass the computed specular color from the fragment evaluation stage to the per-pixel lighting stage.
//  // Varying variable to securely pass the computed specular color from the fragment evaluation stage to the per-pixel lighting stage.
    varying vec3 fragment_stored_clearcoat_specular_color;
//  varying vec3 fragment_stored_clearcoat_specular_color;

    // An optimized mathematical function for computing a base value raised to the fifth power.
//  // An optimized mathematical function for computing a base value raised to the fifth power.
    // This is exclusively utilized for executing Schlick's approximation within the Fresnel equation.
//  // This is exclusively utilized for executing Schlick's approximation within the Fresnel equation.
    float calculate_power_of_five(float base_value) {
//  float calculate_power_of_five(float base_value) {
        float squared_value = base_value * base_value;
//      float squared_value = base_value * base_value;
        return squared_value * squared_value * base_value;
//      return squared_value * squared_value * base_value;
    }
//  }

    void fragment() {
//  void fragment() {
        // 1. Texture Sampling and Base Property Calculation
//      // 1. Texture Sampling and Base Property Calculation
        // Sample the base texture and multiply it by the uniform base color to establish the final substrate coloration.
//      // Sample the base texture and multiply it by the uniform base color to establish the final substrate coloration.
        vec3 sampled_base_color = substrate_base_color * texture(substrate_base_texture, UV * substrate_base_texture_uv_scale).rgb;
//      vec3 sampled_base_color = substrate_base_color * texture(substrate_base_texture, UV * substrate_base_texture_uv_scale).rgb;
        // Sample the clearcoat specular texture and multiply it by the uniform specular color.
//      // Sample the clearcoat specular texture and multiply it by the uniform specular color.
        vec3 sampled_specular_color = clearcoat_specular_color * texture(clearcoat_specular_texture, UV * clearcoat_specular_texture_uv_scale).rgb;
//      vec3 sampled_specular_color = clearcoat_specular_color * texture(clearcoat_specular_texture, UV * clearcoat_specular_texture_uv_scale).rgb;
        // Sample the clearcoat roughness texture and multiply it by the uniform roughness parameter.
//      // Sample the clearcoat roughness texture and multiply it by the uniform roughness parameter.
        float sampled_roughness = clearcoat_roughness * texture(clearcoat_roughness_texture, UV * clearcoat_roughness_texture_uv_scale).r;
//      float sampled_roughness = clearcoat_roughness * texture(clearcoat_roughness_texture, UV * clearcoat_roughness_texture_uv_scale).r;

        // 2. Transmit Base Properties to the Shading Pipeline
//      // 2. Transmit Base Properties to the Shading Pipeline
        // Pass the calculated base color to the rendering engine's built-in ALBEDO parameter.
//      // Pass the calculated base color to the rendering engine's built-in ALBEDO parameter.
        ALBEDO = sampled_base_color;
//      ALBEDO = sampled_base_color;
        // Pass the calculated roughness to the rendering engine's built-in ROUGHNESS parameter.
//      // Pass the calculated roughness to the rendering engine's built-in ROUGHNESS parameter.
        ROUGHNESS = sampled_roughness;
//      ROUGHNESS = sampled_roughness;

        // 3. Normal Map Sampling and Application
//      // 3. Normal Map Sampling and Application
        // Sample the normal map texture utilizing the specified coordinate scaling.
//      // Sample the normal map texture utilizing the specified coordinate scaling.
        vec3 sampled_normal_map_color = texture(substrate_normal_map_texture, UV * substrate_normal_map_texture_uv_scale).rgb;
//      vec3 sampled_normal_map_color = texture(substrate_normal_map_texture, UV * substrate_normal_map_texture_uv_scale).rgb;
        // Apply the sampled normal map to the built-in NORMAL_MAP property to perturb the surface normal.
//      // Apply the sampled normal map to the built-in NORMAL_MAP property to perturb the surface normal.
        NORMAL_MAP = sampled_normal_map_color;
//      NORMAL_MAP = sampled_normal_map_color;
        // Set the intensity of the normal map effect via the NORMAL_MAP_DEPTH property.
//      // Set the intensity of the normal map effect via the NORMAL_MAP_DEPTH property.
        NORMAL_MAP_DEPTH = substrate_normal_map_intensity;
//      NORMAL_MAP_DEPTH = substrate_normal_map_intensity;

        // 4. Preserve Specular Data for Lighting Evaluation
//      // 4. Preserve Specular Data for Lighting Evaluation
        // Store the computed specular color into the varying variable to bypass redundant sampling during the lighting phase.
//      // Store the computed specular color into the varying variable to bypass redundant sampling during the lighting phase.
        fragment_stored_clearcoat_specular_color = sampled_specular_color;
//      fragment_stored_clearcoat_specular_color = sampled_specular_color;
    }
//  }

    void light() {
//  void light() {
        // 1. Core Vector Normalization
//      // 1. Core Vector Normalization
        // Normalize the foundational directional vectors required for physically based lighting equations.
//      // Normalize the foundational directional vectors required for physically based lighting equations.
        vec3 normalized_incident_light_direction = normalize(LIGHT);
//      vec3 normalized_incident_light_direction = normalize(LIGHT);
        vec3 normalized_outgoing_view_direction = normalize(VIEW);
//      vec3 normalized_outgoing_view_direction = normalize(VIEW);
        vec3 normalized_surface_normal_vector = normalize(NORMAL);
//      vec3 normalized_surface_normal_vector = normalize(NORMAL);

        // 2. Directional Dot Product Calculation
//      // 2. Directional Dot Product Calculation
        // Calculate the fundamental dot products, enforcing a strict zero minimum to eliminate negative geometric calculations.
//      // Calculate the fundamental dot products, enforcing a strict zero minimum to eliminate negative geometric calculations.
        float dot_product_normal_and_light_direction = max(0.0, dot(normalized_surface_normal_vector, normalized_incident_light_direction));
//      float dot_product_normal_and_light_direction = max(0.0, dot(normalized_surface_normal_vector, normalized_incident_light_direction));
        float dot_product_normal_and_view_direction = max(0.0, dot(normalized_surface_normal_vector, normalized_outgoing_view_direction));
//      float dot_product_normal_and_view_direction = max(0.0, dot(normalized_surface_normal_vector, normalized_outgoing_view_direction));

        // Ensure that shading is only processed when the surface is illuminated and visible from the current perspective.
//      // Ensure that shading is only processed when the surface is illuminated and visible from the current perspective.
        if (dot_product_normal_and_light_direction > 0.0 && dot_product_normal_and_view_direction > 0.0) {
//      if (dot_product_normal_and_light_direction > 0.0 && dot_product_normal_and_view_direction > 0.0) {
            // Calculate the half-vector, representing the bisector of the light and view directions.
//          // Calculate the half-vector, representing the bisector of the light and view directions.
            vec3 calculated_microfacet_half_vector = normalize(normalized_incident_light_direction + normalized_outgoing_view_direction);
//          vec3 calculated_microfacet_half_vector = normalize(normalized_incident_light_direction + normalized_outgoing_view_direction);

            // Calculate the dot products relative to the half-vector for microfacet evaluation.
//          // Calculate the dot products relative to the half-vector for microfacet evaluation.
            float dot_product_normal_and_half_vector = max(0.0, dot(normalized_surface_normal_vector, calculated_microfacet_half_vector));
//          float dot_product_normal_and_half_vector = max(0.0, dot(normalized_surface_normal_vector, calculated_microfacet_half_vector));
            float dot_product_view_and_half_vector = max(0.0, dot(normalized_outgoing_view_direction, calculated_microfacet_half_vector));
//          float dot_product_view_and_half_vector = max(0.0, dot(normalized_outgoing_view_direction, calculated_microfacet_half_vector));

            // 3. Retrieve Stored Surface Properties
//          // 3. Retrieve Stored Surface Properties
            // Access the material properties established in the fragment stage.
//          // Access the material properties established in the fragment stage.
            vec3 retrieved_substrate_base_color = ALBEDO;
//          vec3 retrieved_substrate_base_color = ALBEDO;
            vec3 retrieved_clearcoat_specular_color = fragment_stored_clearcoat_specular_color;
//          vec3 retrieved_clearcoat_specular_color = fragment_stored_clearcoat_specular_color;
            // Enforce a minimum roughness threshold to prevent mathematical singularities in the GGX distribution.
//          // Enforce a minimum roughness threshold to prevent mathematical singularities in the GGX distribution.
            float retrieved_clearcoat_roughness = max(0.001, ROUGHNESS);
//          float retrieved_clearcoat_roughness = max(0.001, ROUGHNESS);

            // 4. Ashikhmin-Shirley Diffuse Evaluation (Substrate Layer)
//          // 4. Ashikhmin-Shirley Diffuse Evaluation (Substrate Layer)
            // This mathematical model simulates the physical interaction where light penetrating the clearcoat is modulated.
//          // This mathematical model simulates the physical interaction where light penetrating the clearcoat is modulated.
            // It accurately represents the phenomenon where diffuse substrate scattering decreases exponentially at grazing angles,
//          // It accurately represents the phenomenon where diffuse substrate scattering decreases exponentially at grazing angles,
            // precisely because a larger proportion of incident energy is reflected directly by the uppermost clearcoat boundary.
//          // precisely because a larger proportion of incident energy is reflected directly by the uppermost clearcoat boundary.
            vec3 calculated_diffuse_energy_conservation_factor = vec3(1.0) - retrieved_clearcoat_specular_color;
//          vec3 calculated_diffuse_energy_conservation_factor = vec3(1.0) - retrieved_clearcoat_specular_color;
            float calculated_diffuse_directional_light_factor = 1.0 - calculate_power_of_five(1.0 - 0.5 * dot_product_normal_and_light_direction);
//          float calculated_diffuse_directional_light_factor = 1.0 - calculate_power_of_five(1.0 - 0.5 * dot_product_normal_and_light_direction);
            float calculated_diffuse_directional_view_factor = 1.0 - calculate_power_of_five(1.0 - 0.5 * dot_product_normal_and_view_direction);
//          float calculated_diffuse_directional_view_factor = 1.0 - calculate_power_of_five(1.0 - 0.5 * dot_product_normal_and_view_direction);

            // Assemble the final Ashikhmin-Shirley diffuse reflectance component.
//          // Assemble the final Ashikhmin-Shirley diffuse reflectance component.
            vec3 evaluated_diffuse_reflectance_component = (28.0 / (23.0 * PI)) * retrieved_substrate_base_color * calculated_diffuse_energy_conservation_factor * calculated_diffuse_directional_light_factor * calculated_diffuse_directional_view_factor;
//          vec3 evaluated_diffuse_reflectance_component = (28.0 / (23.0 * PI)) * retrieved_substrate_base_color * calculated_diffuse_energy_conservation_factor * calculated_diffuse_directional_light_factor * calculated_diffuse_directional_view_factor;

            // 5. Microfacet Specular Evaluation (Clearcoat Layer)
//          // 5. Microfacet Specular Evaluation (Clearcoat Layer)
            // We implement the Trowbridge-Reitz (GGX) Normal Distribution Function to model the statistical distribution
//          // We implement the Trowbridge-Reitz (GGX) Normal Distribution Function to model the statistical distribution
            // of microfacet orientations, which constructs the clearcoat's highly reflective specular lobe.
//          // of microfacet orientations, which constructs the clearcoat's highly reflective specular lobe.
            float microfacet_roughness_alpha_parameter = retrieved_clearcoat_roughness * retrieved_clearcoat_roughness;
//          float microfacet_roughness_alpha_parameter = retrieved_clearcoat_roughness * retrieved_clearcoat_roughness;
            float microfacet_roughness_alpha_squared = microfacet_roughness_alpha_parameter * microfacet_roughness_alpha_parameter;
//          float microfacet_roughness_alpha_squared = microfacet_roughness_alpha_parameter * microfacet_roughness_alpha_parameter;
            float ggx_distribution_denominator_term = dot_product_normal_and_half_vector * dot_product_normal_and_half_vector * (microfacet_roughness_alpha_squared - 1.0) + 1.0;
//          float ggx_distribution_denominator_term = dot_product_normal_and_half_vector * dot_product_normal_and_half_vector * (microfacet_roughness_alpha_squared - 1.0) + 1.0;
            float ggx_normal_distribution_function_d = microfacet_roughness_alpha_squared / (PI * ggx_distribution_denominator_term * ggx_distribution_denominator_term);
//          float ggx_normal_distribution_function_d = microfacet_roughness_alpha_squared / (PI * ggx_distribution_denominator_term * ggx_distribution_denominator_term);

            // Calculate the Fresnel reflectance (F) utilizing Schlick's computationally efficient approximation.
//          // Calculate the Fresnel reflectance (F) utilizing Schlick's computationally efficient approximation.
            vec3 schlick_fresnel_approximation_f = retrieved_clearcoat_specular_color + (vec3(1.0) - retrieved_clearcoat_specular_color) * calculate_power_of_five(1.0 - dot_product_view_and_half_vector);
//          vec3 schlick_fresnel_approximation_f = retrieved_clearcoat_specular_color + (vec3(1.0) - retrieved_clearcoat_specular_color) * calculate_power_of_five(1.0 - dot_product_view_and_half_vector);

            // The Ashikhmin-Shirley framework integrates a deliberately simplified geometry/denominator term.
//          // The Ashikhmin-Shirley framework integrates a deliberately simplified geometry/denominator term.
            // This design explicitly couples the underlying diffuse lobe with the overlying specular lobe in an elegant manner,
//          // This design explicitly couples the underlying diffuse lobe with the overlying specular lobe in an elegant manner,
            // circumventing the necessity for a highly complex, computationally expensive independent Smith geometry function (G).
//          // circumventing the necessity for a highly complex, computationally expensive independent Smith geometry function (G).
            vec3 evaluated_specular_reflectance_component = (ggx_normal_distribution_function_d * schlick_fresnel_approximation_f) / (4.0 * dot_product_view_and_half_vector * max(dot_product_normal_and_light_direction, dot_product_normal_and_view_direction));
//          vec3 evaluated_specular_reflectance_component = (ggx_normal_distribution_function_d * schlick_fresnel_approximation_f) / (4.0 * dot_product_view_and_half_vector * max(dot_product_normal_and_light_direction, dot_product_normal_and_view_direction));

            // 6. Output Final Evaluated Lighting
//          // 6. Output Final Evaluated Lighting
            // Accumulate the calculated diffuse response into the rendering engine's DIFFUSE_LIGHT output stream,
//          // Accumulate the calculated diffuse response into the rendering engine's DIFFUSE_LIGHT output stream,
            // modulated by the incident light energy and spatial attenuation.
//          // modulated by the incident light energy and spatial attenuation.
            DIFFUSE_LIGHT += evaluated_diffuse_reflectance_component * LIGHT_COLOR * dot_product_normal_and_light_direction * ATTENUATION * direct_light_intensity;
//          DIFFUSE_LIGHT += evaluated_diffuse_reflectance_component * LIGHT_COLOR * dot_product_normal_and_light_direction * ATTENUATION * direct_light_intensity;
            // Accumulate the calculated specular response into the rendering engine's SPECULAR_LIGHT output stream,
//          // Accumulate the calculated specular response into the rendering engine's SPECULAR_LIGHT output stream,
            // correspondingly modulated by the incident light properties.
//          // correspondingly modulated by the incident light properties.
            SPECULAR_LIGHT += evaluated_specular_reflectance_component * LIGHT_COLOR * dot_product_normal_and_light_direction * ATTENUATION * direct_light_intensity;
//          SPECULAR_LIGHT += evaluated_specular_reflectance_component * LIGHT_COLOR * dot_product_normal_and_light_direction * ATTENUATION * direct_light_intensity;
        }
//      }
    }
//  }