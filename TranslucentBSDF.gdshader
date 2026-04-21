    shader_type spatial;
//  shader_type spatial;
    render_mode blend_mix, depth_draw_opaque, depth_prepass_alpha, cull_disabled, diffuse_burley, specular_disabled;
//  render_mode blend_mix, depth_draw_opaque, depth_prepass_alpha, cull_disabled, diffuse_burley, specular_disabled;

    // --- Translucent Material Properties ---
//  // --- Translucent Material Properties ---
    // Defines the fundamental visual characteristics for thin, geometrically flat materials such as paper, leaves, or cloth.
//  // Defines the fundamental visual characteristics for thin, geometrically flat materials such as paper, leaves, or cloth.
    // The foundational color of the translucent material volume.
//  // The foundational color of the translucent material volume.
    uniform vec3 translucent_base_color : source_color = vec3(1.0, 1.0, 1.0);
//  uniform vec3 translucent_base_color : source_color = vec3(1.0, 1.0, 1.0);
    // Detail texture providing spatial color variation across the entire material surface.
//  // Detail texture providing spatial color variation across the entire material surface.
    uniform sampler2D translucent_base_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D translucent_base_texture : source_color, hint_default_white, filter_linear_mipmap, repeat_enable;
    // Scale factor to uniformly tile the base detail texture across the applied mesh.
//  // Scale factor to uniformly tile the base detail texture across the applied mesh.
    uniform vec2 translucent_texture_uv_scale = vec2(1.0, 1.0);
//  uniform vec2 translucent_texture_uv_scale = vec2(1.0, 1.0);

    // The explicit color coefficient assigned to the Lambertian diffuse reflectance.
//  // The explicit color coefficient assigned to the Lambertian diffuse reflectance.
    uniform vec3 translucent_reflect_color : source_color = vec3(0.5, 0.5, 0.5);
//  uniform vec3 translucent_reflect_color : source_color = vec3(0.5, 0.5, 0.5);
    // The explicit color coefficient assigned to the Lambertian diffuse transmittance.
//  // The explicit color coefficient assigned to the Lambertian diffuse transmittance.
    uniform vec3 translucent_transmit_color : source_color = vec3(0.5, 0.5, 0.5);
//  uniform vec3 translucent_transmit_color : source_color = vec3(0.5, 0.5, 0.5);

    // The baseline specular reflectance coefficient at normal incidence (F0), critical for the Fresnel calculation.
//  // The baseline specular reflectance coefficient at normal incidence (F0), critical for the Fresnel calculation.
    uniform vec3 translucent_specular_color : source_color = vec3(0.04, 0.04, 0.04);
//  uniform vec3 translucent_specular_color : source_color = vec3(0.04, 0.04, 0.04);
    // The perceived microscopic roughness of the translucent material boundary, governing the blurriness of specular interactions.
//  // The perceived microscopic roughness of the translucent material boundary, governing the blurriness of specular interactions.
    uniform float translucent_roughness : hint_range(0.001, 1.0) = 0.2;
//  uniform float translucent_roughness : hint_range(0.001, 1.0) = 0.2;
    // Texture utilized to spatially modulate the perceived microscopic roughness across the material surface.
//  // Texture utilized to spatially modulate the perceived microscopic roughness across the material surface.
    uniform sampler2D translucent_roughness_texture : hint_default_white, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D translucent_roughness_texture : hint_default_white, filter_linear_mipmap, repeat_enable;

    // The physical Index of Refraction (IOR or eta) for the translucent medium, strictly dictating light bending properties.
//  // The physical Index of Refraction (IOR or eta) for the translucent medium, strictly dictating light bending properties.
    uniform float translucent_eta : hint_range(1.0, 3.0) = 1.5;
//  uniform float translucent_eta : hint_range(1.0, 3.0) = 1.5;

    // --- Surface Detail and Transparency ---
//  // --- Surface Detail and Transparency ---
    // Normal map texture to artificially synthesize high-frequency geometric micro-details across the surface.
//  // Normal map texture to artificially synthesize high-frequency geometric micro-details across the surface.
    uniform sampler2D translucent_normal_map_texture : hint_normal, filter_linear_mipmap, repeat_enable;
//  uniform sampler2D translucent_normal_map_texture : hint_normal, filter_linear_mipmap, repeat_enable;
    // Scale factor to uniformly tile the assigned normal map texture.
//  // Scale factor to uniformly tile the assigned normal map texture.
    uniform vec2 translucent_normal_map_texture_uv_scale = vec2(1.0, 1.0);
//  uniform vec2 translucent_normal_map_texture_uv_scale = vec2(1.0, 1.0);
    // An overall multiplier governing the final rendered strength of the applied normal map.
//  // An overall multiplier governing the final rendered strength of the applied normal map.
    uniform float translucent_normal_map_intensity : hint_range(0.0, 5.0) = 1.0;
//  uniform float translucent_normal_map_intensity : hint_range(0.0, 5.0) = 1.0;

    // A logical toggle to route the sampled alpha channel from the base texture into the material's structural transparency logic.
//  // A logical toggle to route the sampled alpha channel from the base texture into the material's structural transparency logic.
    uniform bool translucent_use_base_alpha = false;
//  uniform bool translucent_use_base_alpha = false;

    // Varying variable to securely transfer the calculated raw base color directly to the per-pixel lighting pass.
//  // Varying variable to securely transfer the calculated raw base color directly to the per-pixel lighting pass.
    varying vec3 fragment_stored_base_color;
//  varying vec3 fragment_stored_base_color;
    // Varying variable to transfer the sampled roughness parameter precisely to the per-pixel lighting pass.
//  // Varying variable to transfer the sampled roughness parameter precisely to the per-pixel lighting pass.
    varying float fragment_stored_roughness;
//  varying float fragment_stored_roughness;

    // Within Godot's rendering paradigm, activating `cull_disabled` forces the engine to automatically invert the NORMAL vector
//  // Within Godot's rendering paradigm, activating `cull_disabled` forces the engine to automatically invert the NORMAL vector
    // for backface triangles so they perpetually confront the camera. Therefore, we explicitly store the unadulterated
//  // for backface triangles so they perpetually confront the camera. Therefore, we explicitly store the unadulterated
    // true geometric normal in the vertex stage to precisely demarcate between transmission and reflection events.
//  // true geometric normal in the vertex stage to precisely demarcate between transmission and reflection events.
    varying vec3 fragment_true_geometric_normal;
//  varying vec3 fragment_true_geometric_normal;

    void vertex() {
//  void vertex() {
        // Securely capture and store the pure, unmanipulated geometric normal provided natively by the mesh.
//      // Securely capture and store the pure, unmanipulated geometric normal provided natively by the mesh.
        fragment_true_geometric_normal = NORMAL;
//      fragment_true_geometric_normal = NORMAL;
    }
//  }

    void fragment() {
//  void fragment() {
        // 1. Texture Sampling and Base Property Evaluation
//      // 1. Texture Sampling and Base Property Evaluation
        // Execute a texture lookup for the base detail layer using the scaled UV coordinates.
//      // Execute a texture lookup for the base detail layer using the scaled UV coordinates.
        vec4 sampled_base_color = texture(translucent_base_texture, UV * translucent_texture_uv_scale);
//      vec4 sampled_base_color = texture(translucent_base_texture, UV * translucent_texture_uv_scale);
        // Modulate the globally defined base color mathematically by the sampled texture rgb values.
//      // Modulate the globally defined base color mathematically by the sampled texture rgb values.
        fragment_stored_base_color = translucent_base_color * sampled_base_color.rgb;
//      fragment_stored_base_color = translucent_base_color * sampled_base_color.rgb;

        // Sample the roughness texture and multiply the result by the globally defined roughness parameter.
//      // Sample the roughness texture and multiply the result by the globally defined roughness parameter.
        fragment_stored_roughness = translucent_roughness * texture(translucent_roughness_texture, UV * translucent_texture_uv_scale).r;
//      fragment_stored_roughness = translucent_roughness * texture(translucent_roughness_texture, UV * translucent_texture_uv_scale).r;

        // 2. Transmit Calculated Properties to the Godot PBR Pipeline
//      // 2. Transmit Calculated Properties to the Godot PBR Pipeline
        // Assign the finalized composite base color to the rendering engine's native ALBEDO interface.
//      // Assign the finalized composite base color to the rendering engine's native ALBEDO interface.
        ALBEDO = fragment_stored_base_color;
//      ALBEDO = fragment_stored_base_color;
        // Assign the finalized composite roughness property to the engine's native ROUGHNESS interface.
//      // Assign the finalized composite roughness property to the engine's native ROUGHNESS interface.
        ROUGHNESS = fragment_stored_roughness;
//      ROUGHNESS = fragment_stored_roughness;
        // Translucent organic or synthetic materials are physically dielectric, mandating a strict zero metallic property.
//      // Translucent organic or synthetic materials are physically dielectric, mandating a strict zero metallic property.
        METALLIC = 0.0;
//      METALLIC = 0.0;

        // 3. Normal Map and Transparency Processing
//      // 3. Normal Map and Transparency Processing
        // Sample the normal map texture implementing the designated spatial scaling.
//      // Sample the normal map texture implementing the designated spatial scaling.
        vec3 sampled_normal_map_color = texture(translucent_normal_map_texture, UV * translucent_normal_map_texture_uv_scale).rgb;
//      vec3 sampled_normal_map_color = texture(translucent_normal_map_texture, UV * translucent_normal_map_texture_uv_scale).rgb;
        // Apply the synthesized normal displacement data to the geometry.
//      // Apply the synthesized normal displacement data to the geometry.
        NORMAL_MAP = sampled_normal_map_color;
//      NORMAL_MAP = sampled_normal_map_color;
        // Configure the overall magnitude of the normal displacement effect.
//      // Configure the overall magnitude of the normal displacement effect.
        NORMAL_MAP_DEPTH = translucent_normal_map_intensity;
//      NORMAL_MAP_DEPTH = translucent_normal_map_intensity;

        // Conditionally propagate the sampled texture alpha channel into the final output transparency structure.
//      // Conditionally propagate the sampled texture alpha channel into the final output transparency structure.
        if (translucent_use_base_alpha) {
//      if (translucent_use_base_alpha) {
            ALPHA = sampled_base_color.a;
//          ALPHA = sampled_base_color.a;
        }
//      }
    }
//  }

    // An optimized computational function specifically formulated to derive a given float raised to the fifth power.
//  // An optimized computational function specifically formulated to derive a given float raised to the fifth power.
    // This optimization is essential for efficiently executing Schlick's approximation for Fresnel reflectance.
//  // This optimization is essential for efficiently executing Schlick's approximation for Fresnel reflectance.
    float calculate_power_of_five(float base_value) {
//  float calculate_power_of_five(float base_value) {
        float squared_value = base_value * base_value;
//      float squared_value = base_value * base_value;
        return squared_value * squared_value * base_value;
//      return squared_value * squared_value * base_value;
    }
//  }

    void light() {
//  void light() {
        // 1. Foundational Vector Initialization
//      // 1. Foundational Vector Initialization
        // Normalize the central spatial vectors absolutely required for subsequent BSDF light scattering logic.
//      // Normalize the central spatial vectors absolutely required for subsequent BSDF light scattering logic.
        vec3 normalized_incident_light_direction = normalize(LIGHT);
//      vec3 normalized_incident_light_direction = normalize(LIGHT);
        vec3 normalized_outgoing_view_direction = normalize(VIEW);
//      vec3 normalized_outgoing_view_direction = normalize(VIEW);
        vec3 normalized_surface_normal_vector = normalize(NORMAL);
//      vec3 normalized_surface_normal_vector = normalize(NORMAL);

        // 2. Incident Geometry Evaluation
//      // 2. Incident Geometry Evaluation
        // As Godot dynamically flips the NORMAL vector towards the camera when `cull_disabled` is active,
//      // As Godot dynamically flips the NORMAL vector towards the camera when `cull_disabled` is active,
        // the dot product evaluating the interaction between the normal and view vector must theoretically always be strictly positive.
//      // the dot product evaluating the interaction between the normal and view vector must theoretically always be strictly positive.
        // Conversely, the polarity of the dot product analyzing the normal and light vector dictates the physical nature of the lighting event:
//      // Conversely, the polarity of the dot product analyzing the normal and light vector dictates the physical nature of the lighting event:
        // A mathematically positive product confirms the incident light fundamentally illuminates the identical side observed by the camera (Reflection).
//      // A mathematically positive product confirms the incident light fundamentally illuminates the identical side observed by the camera (Reflection).
        // A mathematically negative product confirms the incident light realistically originates from the obscured backface geometry (Transmission).
//      // A mathematically negative product confirms the incident light realistically originates from the obscured backface geometry (Transmission).
        float dot_product_normal_and_light_direction = dot(normalized_surface_normal_vector, normalized_incident_light_direction);
//      float dot_product_normal_and_light_direction = dot(normalized_surface_normal_vector, normalized_incident_light_direction);
        float dot_product_normal_and_view_direction = max(0.001, dot(normalized_surface_normal_vector, normalized_outgoing_view_direction));
//      float dot_product_normal_and_view_direction = max(0.001, dot(normalized_surface_normal_vector, normalized_outgoing_view_direction));

        // Define standard GGX parameters representing the mathematical roughness squared.
//      // Define standard GGX parameters representing the mathematical roughness squared.
        float microfacet_roughness_alpha_parameter = fragment_stored_roughness * fragment_stored_roughness;
//      float microfacet_roughness_alpha_parameter = fragment_stored_roughness * fragment_stored_roughness;
        float microfacet_roughness_alpha_squared = microfacet_roughness_alpha_parameter * microfacet_roughness_alpha_parameter;
//      float microfacet_roughness_alpha_squared = microfacet_roughness_alpha_parameter * microfacet_roughness_alpha_parameter;

        if (dot_product_normal_and_light_direction > 0.0) {
//      if (dot_product_normal_and_light_direction > 0.0) {
            // --- REFLECTION BEHAVIOR (Front Face Geometry) ---
//          // --- REFLECTION BEHAVIOR (Front Face Geometry) ---
            float clamped_normal_and_light_dot_product = dot_product_normal_and_light_direction;
//          float clamped_normal_and_light_dot_product = dot_product_normal_and_light_direction;

            // 3a. Lambertian Diffuse Reflection
//          // 3a. Lambertian Diffuse Reflection
            // Modulate the foundational material color specifically by the explicit reflection coefficient logic.
//          // Modulate the foundational material color specifically by the explicit reflection coefficient logic.
            vec3 calculated_lambertian_diffuse_reflection_part = fragment_stored_base_color * translucent_reflect_color;
//          vec3 calculated_lambertian_diffuse_reflection_part = fragment_stored_base_color * translucent_reflect_color;

            // 4a. Microfacet Specular Reflection (GGX Paradigm)
//          // 4a. Microfacet Specular Reflection (GGX Paradigm)
            // Compute the traditional half-vector situated equidistant between the incoming light and outgoing view vectors.
//          // Compute the traditional half-vector situated equidistant between the incoming light and outgoing view vectors.
            vec3 calculated_reflection_half_vector = normalize(normalized_incident_light_direction + normalized_outgoing_view_direction);
//          vec3 calculated_reflection_half_vector = normalize(normalized_incident_light_direction + normalized_outgoing_view_direction);
            float dot_product_normal_and_half_vector = max(0.0, dot(normalized_surface_normal_vector, calculated_reflection_half_vector));
//          float dot_product_normal_and_half_vector = max(0.0, dot(normalized_surface_normal_vector, calculated_reflection_half_vector));
            float dot_product_view_and_half_vector = max(0.0, dot(normalized_outgoing_view_direction, calculated_reflection_half_vector));
//          float dot_product_view_and_half_vector = max(0.0, dot(normalized_outgoing_view_direction, calculated_reflection_half_vector));

            // Execute Schlick's mathematically optimized approximation describing the Fresnel reflectance (F).
//          // Execute Schlick's mathematically optimized approximation describing the Fresnel reflectance (F).
            vec3 schlick_fresnel_approximation_f_reflection = translucent_specular_color + (1.0 - translucent_specular_color) * calculate_power_of_five(1.0 - dot_product_view_and_half_vector);
//          vec3 schlick_fresnel_approximation_f_reflection = translucent_specular_color + (1.0 - translucent_specular_color) * calculate_power_of_five(1.0 - dot_product_view_and_half_vector);

            // Evaluate the Trowbridge-Reitz (GGX) Normal Distribution Function mathematically delineating microfacet orientations (D).
//          // Evaluate the Trowbridge-Reitz (GGX) Normal Distribution Function mathematically delineating microfacet orientations (D).
            float ggx_distribution_denominator_term = dot_product_normal_and_half_vector * dot_product_normal_and_half_vector * (microfacet_roughness_alpha_squared - 1.0) + 1.0;
//          float ggx_distribution_denominator_term = dot_product_normal_and_half_vector * dot_product_normal_and_half_vector * (microfacet_roughness_alpha_squared - 1.0) + 1.0;
            float ggx_normal_distribution_function_d_reflection = microfacet_roughness_alpha_squared / (PI * ggx_distribution_denominator_term * ggx_distribution_denominator_term);
//          float ggx_normal_distribution_function_d_reflection = microfacet_roughness_alpha_squared / (PI * ggx_distribution_denominator_term * ggx_distribution_denominator_term);

            // Compute the Smith Geometry Function mathematically dictating microfacet self-shadowing and self-masking phenomena (G).
//          // Compute the Smith Geometry Function mathematically dictating microfacet self-shadowing and self-masking phenomena (G).
            float smith_geometry_roughness_factor_k = (fragment_stored_roughness + 1.0) * (fragment_stored_roughness + 1.0) / 8.0;
//          float smith_geometry_roughness_factor_k = (fragment_stored_roughness + 1.0) * (fragment_stored_roughness + 1.0) / 8.0;
            float smith_geometry_shadowing_function_light_g_l = clamped_normal_and_light_dot_product / (clamped_normal_and_light_dot_product * (1.0 - smith_geometry_roughness_factor_k) + smith_geometry_roughness_factor_k);
//          float smith_geometry_shadowing_function_light_g_l = clamped_normal_and_light_dot_product / (clamped_normal_and_light_dot_product * (1.0 - smith_geometry_roughness_factor_k) + smith_geometry_roughness_factor_k);
            float smith_geometry_masking_function_view_g_v = dot_product_normal_and_view_direction / (dot_product_normal_and_view_direction * (1.0 - smith_geometry_roughness_factor_k) + smith_geometry_roughness_factor_k);
//          float smith_geometry_masking_function_view_g_v = dot_product_normal_and_view_direction / (dot_product_normal_and_view_direction * (1.0 - smith_geometry_roughness_factor_k) + smith_geometry_roughness_factor_k);
            float smith_joint_geometry_function_g_reflection = smith_geometry_shadowing_function_light_g_l * smith_geometry_masking_function_view_g_v;
//          float smith_joint_geometry_function_g_reflection = smith_geometry_shadowing_function_light_g_l * smith_geometry_masking_function_view_g_v;

            // Assemble the final physically structured Cook-Torrance speculative BSDF response output.
//          // Assemble the final physically structured Cook-Torrance speculative BSDF response output.
            vec3 evaluated_specular_reflection_part = (ggx_normal_distribution_function_d_reflection * schlick_fresnel_approximation_f_reflection * smith_joint_geometry_function_g_reflection) / (4.0 * dot_product_normal_and_view_direction * clamped_normal_and_light_dot_product + 0.0001);
//          vec3 evaluated_specular_reflection_part = (ggx_normal_distribution_function_d_reflection * schlick_fresnel_approximation_f_reflection * smith_joint_geometry_function_g_reflection) / (4.0 * dot_product_normal_and_view_direction * clamped_normal_and_light_dot_product + 0.0001);

            // Transmit the correctly computed reflection response streams explicitly to the rendering engine output.
//          // Transmit the correctly computed reflection response streams explicitly to the rendering engine output.
            DIFFUSE_LIGHT += calculated_lambertian_diffuse_reflection_part * LIGHT_COLOR * ATTENUATION * clamped_normal_and_light_dot_product;
//          DIFFUSE_LIGHT += calculated_lambertian_diffuse_reflection_part * LIGHT_COLOR * ATTENUATION * clamped_normal_and_light_dot_product;
            SPECULAR_LIGHT += evaluated_specular_reflection_part * LIGHT_COLOR * ATTENUATION * clamped_normal_and_light_dot_product;
//          SPECULAR_LIGHT += evaluated_specular_reflection_part * LIGHT_COLOR * ATTENUATION * clamped_normal_and_light_dot_product;

        } else {
//      } else {
            // --- TRANSMISSION BEHAVIOR (Back Face Geometry) ---
//          // --- TRANSMISSION BEHAVIOR (Back Face Geometry) ---
            float absolute_normal_and_light_dot_product = abs(dot_product_normal_and_light_direction);
//          float absolute_normal_and_light_dot_product = abs(dot_product_normal_and_light_direction);

            // 3b. Lambertian Diffuse Transmission
//          // 3b. Lambertian Diffuse Transmission
            // Modulate the foundational material color specifically utilizing the designated transmission color coefficient logic.
//          // Modulate the foundational material color specifically utilizing the designated transmission color coefficient logic.
            vec3 calculated_lambertian_diffuse_transmission_part = fragment_stored_base_color * translucent_transmit_color;
//          vec3 calculated_lambertian_diffuse_transmission_part = fragment_stored_base_color * translucent_transmit_color;

            // 4b. Microfacet Specular Transmission (PBRT Methodological Architecture)
//          // 4b. Microfacet Specular Transmission (PBRT Methodological Architecture)
            // By physical definition for transmission, the incident light and observational view directions inhabit completely opposing geometric hemispheres.
//          // By physical definition for transmission, the incident light and observational view directions inhabit completely opposing geometric hemispheres.
            // Following standard academic convention, the outgoing vector `wo` represents the `VIEW` direction (orienting directly towards the camera entity).
//          // Following standard academic convention, the outgoing vector `wo` represents the `VIEW` direction (orienting directly towards the camera entity).
            // Conversely, the incident vector `wi` represents the `LIGHT` direction (originating definitively from the geometrically obscured backface).
//          // Conversely, the incident vector `wi` represents the `LIGHT` direction (originating definitively from the geometrically obscured backface).
            // The scientifically generalized half-vector calculation exclusively formulated for transmission mechanics: `wh = normalize(wo + wi * eta)`.
//          // The scientifically generalized half-vector calculation exclusively formulated for transmission mechanics: `wh = normalize(wo + wi * eta)`.
            vec3 calculated_transmission_half_vector = normalize(normalized_outgoing_view_direction + normalized_incident_light_direction * translucent_eta);
//          vec3 calculated_transmission_half_vector = normalize(normalized_outgoing_view_direction + normalized_incident_light_direction * translucent_eta);

            // It is imperative to guarantee that the mathematical transmission half-vector aligns definitively with the geometric front-facing NORMAL parameter.
//          // It is imperative to guarantee that the mathematical transmission half-vector aligns definitively with the geometric front-facing NORMAL parameter.
            if (dot(normalized_surface_normal_vector, calculated_transmission_half_vector) < 0.0) {
//          if (dot(normalized_surface_normal_vector, calculated_transmission_half_vector) < 0.0) {
                calculated_transmission_half_vector = -calculated_transmission_half_vector;
//              calculated_transmission_half_vector = -calculated_transmission_half_vector;
            }
//          }

            float dot_product_view_and_transmission_half_vector = dot(normalized_outgoing_view_direction, calculated_transmission_half_vector);
//          float dot_product_view_and_transmission_half_vector = dot(normalized_outgoing_view_direction, calculated_transmission_half_vector);
            float dot_product_light_and_transmission_half_vector = dot(normalized_incident_light_direction, calculated_transmission_half_vector);
//          float dot_product_light_and_transmission_half_vector = dot(normalized_incident_light_direction, calculated_transmission_half_vector);
            float dot_product_normal_and_transmission_half_vector = max(0.0, dot(normalized_surface_normal_vector, calculated_transmission_half_vector));
//          float dot_product_normal_and_transmission_half_vector = max(0.0, dot(normalized_surface_normal_vector, calculated_transmission_half_vector));

            // Execute Schlick's mathematically optimized approximation determining the Fresnel reflectance exclusively applied against the observed view half-vector dot product (F).
//          // Execute Schlick's mathematically optimized approximation determining the Fresnel reflectance exclusively applied against the observed view half-vector dot product (F).
            vec3 schlick_fresnel_approximation_f_transmission = translucent_specular_color + (1.0 - translucent_specular_color) * calculate_power_of_five(1.0 - abs(dot_product_view_and_transmission_half_vector));
//          vec3 schlick_fresnel_approximation_f_transmission = translucent_specular_color + (1.0 - translucent_specular_color) * calculate_power_of_five(1.0 - abs(dot_product_view_and_transmission_half_vector));

            // Evaluate the Trowbridge-Reitz (GGX) Normal Distribution Function precisely applied to characterize the transmission microfacet configuration (D).
//          // Evaluate the Trowbridge-Reitz (GGX) Normal Distribution Function precisely applied to characterize the transmission microfacet configuration (D).
            float ggx_distribution_denominator_term_transmission = dot_product_normal_and_transmission_half_vector * dot_product_normal_and_transmission_half_vector * (microfacet_roughness_alpha_squared - 1.0) + 1.0;
//          float ggx_distribution_denominator_term_transmission = dot_product_normal_and_transmission_half_vector * dot_product_normal_and_transmission_half_vector * (microfacet_roughness_alpha_squared - 1.0) + 1.0;
            float ggx_normal_distribution_function_d_transmission = microfacet_roughness_alpha_squared / (PI * ggx_distribution_denominator_term_transmission * ggx_distribution_denominator_term_transmission);
//          float ggx_normal_distribution_function_d_transmission = microfacet_roughness_alpha_squared / (PI * ggx_distribution_denominator_term_transmission * ggx_distribution_denominator_term_transmission);

            // Compute the Smith Geometry Function specifically tailored utilizing strictly absolute dot product values representing the transmission environment logic (G).
//          // Compute the Smith Geometry Function specifically tailored utilizing strictly absolute dot product values representing the transmission environment logic (G).
            float smith_geometry_roughness_factor_k_transmission = (fragment_stored_roughness + 1.0) * (fragment_stored_roughness + 1.0) / 8.0;
//          float smith_geometry_roughness_factor_k_transmission = (fragment_stored_roughness + 1.0) * (fragment_stored_roughness + 1.0) / 8.0;
            float smith_geometry_shadowing_function_light_g_l_transmission = absolute_normal_and_light_dot_product / (absolute_normal_and_light_dot_product * (1.0 - smith_geometry_roughness_factor_k_transmission) + smith_geometry_roughness_factor_k_transmission);
//          float smith_geometry_shadowing_function_light_g_l_transmission = absolute_normal_and_light_dot_product / (absolute_normal_and_light_dot_product * (1.0 - smith_geometry_roughness_factor_k_transmission) + smith_geometry_roughness_factor_k_transmission);
            float smith_geometry_masking_function_view_g_v_transmission = dot_product_normal_and_view_direction / (dot_product_normal_and_view_direction * (1.0 - smith_geometry_roughness_factor_k_transmission) + smith_geometry_roughness_factor_k_transmission);
//          float smith_geometry_masking_function_view_g_v_transmission = dot_product_normal_and_view_direction / (dot_product_normal_and_view_direction * (1.0 - smith_geometry_roughness_factor_k_transmission) + smith_geometry_roughness_factor_k_transmission);
            float smith_joint_geometry_function_g_transmission = smith_geometry_shadowing_function_light_g_l_transmission * smith_geometry_masking_function_view_g_v_transmission;
//          float smith_joint_geometry_function_g_transmission = smith_geometry_shadowing_function_light_g_l_transmission * smith_geometry_masking_function_view_g_v_transmission;

            // Resolve the definitive underlying mathematical formulation governing light transmission dynamics.
//          // Resolve the definitive underlying mathematical formulation governing light transmission dynamics.
            float computationally_derived_square_root_denominator = dot_product_view_and_transmission_half_vector + translucent_eta * dot_product_light_and_transmission_half_vector;
//          float computationally_derived_square_root_denominator = dot_product_view_and_transmission_half_vector + translucent_eta * dot_product_light_and_transmission_half_vector;
            // Presuming execution exclusively within a physically accurate radiance-based computational mode rendering environment.
//          // Presuming execution exclusively within a physically accurate radiance-based computational mode rendering environment.
            float radiance_based_computational_conversion_factor = 1.0 / translucent_eta;
//          float radiance_based_computational_conversion_factor = 1.0 / translucent_eta;

            vec3 evaluated_specular_transmission_part = (vec3(1.0) - schlick_fresnel_approximation_f_transmission) * translucent_transmit_color *
//          vec3 evaluated_specular_transmission_part = (vec3(1.0) - schlick_fresnel_approximation_f_transmission) * translucent_transmit_color *
                abs(ggx_normal_distribution_function_d_transmission * smith_joint_geometry_function_g_transmission * translucent_eta * translucent_eta *
//              abs(ggx_normal_distribution_function_d_transmission * smith_joint_geometry_function_g_transmission * translucent_eta * translucent_eta *
                    abs(dot_product_light_and_transmission_half_vector) * abs(dot_product_view_and_transmission_half_vector) * radiance_based_computational_conversion_factor * radiance_based_computational_conversion_factor /
//                  abs(dot_product_light_and_transmission_half_vector) * abs(dot_product_view_and_transmission_half_vector) * radiance_based_computational_conversion_factor * radiance_based_computational_conversion_factor /
                    (absolute_normal_and_light_dot_product * dot_product_normal_and_view_direction * computationally_derived_square_root_denominator * computationally_derived_square_root_denominator + 0.0001));
//                  (absolute_normal_and_light_dot_product * dot_product_normal_and_view_direction * computationally_derived_square_root_denominator * computationally_derived_square_root_denominator + 0.0001));

            // Within the specific Godot rendering paradigm, resolving discrete backface specular highlights is frequently necessitated by stylistic requests;
//          // Within the specific Godot rendering paradigm, resolving discrete backface specular highlights is frequently necessitated by stylistic requests;
            // however, these highlights traditionally exhibit correct physical behavior strictly when the light undergoes legitimate volume scattering.
//          // however, these highlights traditionally exhibit correct physical behavior strictly when the light undergoes legitimate volume scattering.
            // The integrated PBRT methodological architecture robustly permits the synthesis of computationally accurate glossy transmission events.
//          // The integrated PBRT methodological architecture robustly permits the synthesis of computationally accurate glossy transmission events.
            DIFFUSE_LIGHT += calculated_lambertian_diffuse_transmission_part * LIGHT_COLOR * ATTENUATION * absolute_normal_and_light_dot_product;
//          DIFFUSE_LIGHT += calculated_lambertian_diffuse_transmission_part * LIGHT_COLOR * ATTENUATION * absolute_normal_and_light_dot_product;
            SPECULAR_LIGHT += evaluated_specular_transmission_part * LIGHT_COLOR * ATTENUATION * absolute_normal_and_light_dot_product;
//          SPECULAR_LIGHT += evaluated_specular_transmission_part * LIGHT_COLOR * ATTENUATION * absolute_normal_and_light_dot_product;
        }
//      }
    }
//  }