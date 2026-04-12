    shader_type spatial;
//  shader_type spatial;

    // We allow standard ambient lighting so Godot can process Skybox and Ambient IBL
//  // We allow standard ambient lighting so Godot can process Skybox and Ambient IBL
    // render_mode ambient_light_disabled;
//  // render_mode ambient_light_disabled;

    // ---------------------------------------------------------------------------------
//  // ---------------------------------------------------------------------------------
    // UNIFORMS
//  // UNIFORMS
    // ---------------------------------------------------------------------------------
//  // ---------------------------------------------------------------------------------
    // Important: When you import the MERL EXR atlas as a Texture3D in Godot,
//  // Important: When you import the MERL EXR atlas as a Texture3D in Godot,
    // make sure the Filter is set to "Linear" and Repeat is "Disabled"
//  // make sure the Filter is set to "Linear" and Repeat is "Disabled"
    // This should be imported in Godot as a 3D Texture with dimensions 180x90x90 (10 columns, 9 rows).
//  // This should be imported in Godot as a 3D Texture with dimensions 180x90x90 (10 columns, 9 rows).
    uniform sampler3D merl_brdf_texture_atlas : source_color, filter_linear_mipmap, repeat_disable;
//  uniform sampler3D merl_brdf_texture_atlas : source_color, filter_linear_mipmap, repeat_disable;

    uniform vec3 base_color_albedo : source_color = vec3(1.0, 1.0, 1.0);
//  uniform vec3 base_color_albedo : source_color = vec3(1.0, 1.0, 1.0);
    uniform float exposure_multiplier = 1.0;
//  uniform float exposure_multiplier = 1.0;

    // We expose roughness and metallic for ambient/skybox reflections since MERL BRDF doesn't handle them
//  // We expose roughness and metallic for ambient/skybox reflections since MERL BRDF doesn't handle them
    uniform float environment_roughness : hint_range(0.0, 1.0) = 1.0;
//  uniform float environment_roughness : hint_range(0.0, 1.0) = 1.0;
    uniform float environment_metallic : hint_range(0.0, 1.0) = 0.0;
//  uniform float environment_metallic : hint_range(0.0, 1.0) = 0.0;

    // ---------------------------------------------------------------------------------
//  // ---------------------------------------------------------------------------------
    // FRAGMENT
//  // FRAGMENT
    // ---------------------------------------------------------------------------------
//  // ---------------------------------------------------------------------------------
    void fragment() {
//  void fragment() {
        // We set ALBEDO, METALLIC, and ROUGHNESS so Godot can compute built-in Ambient and Skybox Image-Based Lighting
//      // We set ALBEDO, METALLIC, and ROUGHNESS so Godot can compute built-in Ambient and Skybox Image-Based Lighting
        // The light() function still completely overrides the direct lighting calculation with MERL
//      // The light() function still completely overrides the direct lighting calculation with MERL
        ALBEDO = base_color_albedo;
//      ALBEDO = base_color_albedo;
        METALLIC = environment_metallic;
//      METALLIC = environment_metallic;
        ROUGHNESS = environment_roughness;
//      ROUGHNESS = environment_roughness;
    }
//  }

    // ---------------------------------------------------------------------------------
//  // ---------------------------------------------------------------------------------
    // CUSTOM LIGHTING FUNCTION (MERL BRDF)
//  // CUSTOM LIGHTING FUNCTION (MERL BRDF)
    // ---------------------------------------------------------------------------------
//  // ---------------------------------------------------------------------------------
    void light() {
//  void light() {
        // Normal, Light, and View vectors (all normalized by Godot automatically)
//      // Normal, Light, and View vectors (all normalized by Godot automatically)
        vec3 light_direction = LIGHT;
//      vec3 light_direction = LIGHT;
        vec3 view_direction = VIEW;
//      vec3 view_direction = VIEW;
        vec3 surface_normal = NORMAL;
//      vec3 surface_normal = NORMAL;

        // 1. Calculate Half Vector
//      // 1. Calculate Half Vector
        vec3 half_vector = normalize(light_direction + view_direction);
//      vec3 half_vector = normalize(light_direction + view_direction);

        // 2. Theta_Half (Angle between Normal and Half vector)
//      // 2. Theta_Half (Angle between Normal and Half vector)
        float cosine_angle_half = clamp(dot(surface_normal, half_vector), 0.0, 1.0);
//      float cosine_angle_half = clamp(dot(surface_normal, half_vector), 0.0, 1.0);
        float angle_theta_half = acos(cosine_angle_half);
//      float angle_theta_half = acos(cosine_angle_half);

        // 3. Theta_Diff (Angle between Light and Half vector)
//      // 3. Theta_Diff (Angle between Light and Half vector)
        float cosine_angle_diff = clamp(dot(light_direction, half_vector), 0.0, 1.0);
//      float cosine_angle_diff = clamp(dot(light_direction, half_vector), 0.0, 1.0);
        float angle_theta_diff = acos(cosine_angle_diff);
//      float angle_theta_diff = acos(cosine_angle_diff);

        // 4. Phi_Diff (Azimuthal angle difference)
//      // 4. Phi_Diff (Azimuthal angle difference)
        // We build an orthonormal basis (X, Y) around the Normal and Half vectors
//      // We build an orthonormal basis (X, Y) around the Normal and Half vectors
        // to project the Light vector onto it.
//      // to project the Light vector onto it.
        vec3 binormal_y_axis = cross(surface_normal, half_vector);
//      vec3 binormal_y_axis = cross(surface_normal, half_vector);
        float binormal_length = length(binormal_y_axis);
//      float binormal_length = length(binormal_y_axis);

        if (binormal_length < 1e-5) {
//      if (binormal_length < 1e-5) {
            // Fallback if N and H are completely parallel
//          // Fallback if N and H are completely parallel
            binormal_y_axis = cross(surface_normal, vec3(1.0, 0.0, 0.0));
//          binormal_y_axis = cross(surface_normal, vec3(1.0, 0.0, 0.0));
            if (length(binormal_y_axis) < 1e-5) {
//          if (length(binormal_y_axis) < 1e-5) {
                binormal_y_axis = cross(surface_normal, vec3(0.0, 1.0, 0.0));
//              binormal_y_axis = cross(surface_normal, vec3(0.0, 1.0, 0.0));
            }
//          }
            binormal_y_axis = normalize(binormal_y_axis);
//          binormal_y_axis = normalize(binormal_y_axis);
        } else {
//      } else {
            binormal_y_axis = binormal_y_axis / binormal_length;
//          binormal_y_axis = binormal_y_axis / binormal_length;
        }
//      }

        vec3 tangent_x_axis = normalize(cross(binormal_y_axis, half_vector));
//      vec3 tangent_x_axis = normalize(cross(binormal_y_axis, half_vector));

        float light_projected_x = dot(light_direction, tangent_x_axis);
//      float light_projected_x = dot(light_direction, tangent_x_axis);
        float light_projected_y = dot(light_direction, binormal_y_axis);
//      float light_projected_y = dot(light_direction, binormal_y_axis);

        float angle_phi_diff = atan(light_projected_y, light_projected_x);
//      float angle_phi_diff = atan(light_projected_y, light_projected_x);
        if (angle_phi_diff < 0.0) {
//      if (angle_phi_diff < 0.0) {
            angle_phi_diff += PI; // Because of reciprocity, BRDF is unchanged under phi_d + PI
//          angle_phi_diff += PI; // Because of reciprocity, BRDF is unchanged under phi_d + PI
        }
//      }

        // 5. Map to 3D Texture Coordinates (U, V, W)
//      // 5. Map to 3D Texture Coordinates (U, V, W)
        // W (Depth axis) = Theta_H (MERL uses a non-linear mapping for Theta_H)
//      // W (Depth axis) = Theta_H (MERL uses a non-linear mapping for Theta_H)
        float texture_depth_w = sqrt(angle_theta_half / (PI * 0.5));
//      float texture_depth_w = sqrt(angle_theta_half / (PI * 0.5));

        // V (Y axis) = Theta_D (Linear mapping)
//      // V (Y axis) = Theta_D (Linear mapping)
        float texture_coord_v = angle_theta_diff / (PI * 0.5);
//      float texture_coord_v = angle_theta_diff / (PI * 0.5);

        // U (X axis) = Phi_D (Linear mapping)
//      // U (X axis) = Phi_D (Linear mapping)
        float texture_coord_u = angle_phi_diff / PI;
//      float texture_coord_u = angle_phi_diff / PI;

        // Clamp coordinates to ensure we don't sample outside the 3D texture bounds
//      // Clamp coordinates to ensure we don't sample outside the 3D texture bounds
        vec3 texture_coordinates_3d = vec3(clamp(texture_coord_u, 0.0, 1.0), clamp(texture_coord_v, 0.0, 1.0), clamp(texture_depth_w, 0.0, 1.0));
//      vec3 texture_coordinates_3d = vec3(clamp(texture_coord_u, 0.0, 1.0), clamp(texture_coord_v, 0.0, 1.0), clamp(texture_depth_w, 0.0, 1.0));

        // Sample the MERL 3D Texture
//      // Sample the MERL 3D Texture
        vec3 sampled_brdf_color = texture(merl_brdf_texture_atlas, texture_coordinates_3d).rgb;
//      vec3 sampled_brdf_color = texture(merl_brdf_texture_atlas, texture_coordinates_3d).rgb;

        // The Rendering Equation: Radiance = BRDF * Light_Color * Attenuation * NdotL
//      // The Rendering Equation: Radiance = BRDF * Light_Color * Attenuation * NdotL
        float normal_dot_light = max(dot(surface_normal, light_direction), 0.0);
//      float normal_dot_light = max(dot(surface_normal, light_direction), 0.0);

        // We assign it to SPECULAR_LIGHT to override standard PBR handling.
//      // We assign it to SPECULAR_LIGHT to override standard PBR handling.
        // (Note: MERL contains both diffuse and specular reflections baked together)
//      // (Note: MERL contains both diffuse and specular reflections baked together)
        SPECULAR_LIGHT += sampled_brdf_color * base_color_albedo * LIGHT_COLOR * ATTENUATION * normal_dot_light * exposure_multiplier;
//      SPECULAR_LIGHT += sampled_brdf_color * base_color_albedo * LIGHT_COLOR * ATTENUATION * normal_dot_light * exposure_multiplier;
    }
//  }
