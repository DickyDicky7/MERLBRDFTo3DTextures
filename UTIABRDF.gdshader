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
    // Important: When you import the UTIA BRDF PG14 EXR texture in Godot,
//  // Important: When you import the UTIA BRDF PG14 EXR texture in Godot,
    // make sure the Filter is set to "Linear" and Repeat is "Disabled"
//  // make sure the Filter is set to "Linear" and Repeat is "Disabled"
    uniform sampler2D utia_brdf_texture : source_color, filter_linear, repeat_disable;
//  uniform sampler2D utia_brdf_texture : source_color, filter_linear, repeat_disable;

    uniform vec3 base_color_albedo : source_color = vec3(1.0, 1.0, 1.0);
//  uniform vec3 base_color_albedo : source_color = vec3(1.0, 1.0, 1.0);
    uniform float exposure_multiplier = 1.0;
//  uniform float exposure_multiplier = 1.0;

    // We expose roughness and metallic for ambient/skybox reflections since UTIA BRDF doesn't handle them
//  // We expose roughness and metallic for ambient/skybox reflections since UTIA BRDF doesn't handle them
    uniform float environment_roughness : hint_range(0.0, 1.0) = 1.0;
//  uniform float environment_roughness : hint_range(0.0, 1.0) = 1.0;
    uniform float environment_metallic : hint_range(0.0, 1.0) = 0.0;
//  uniform float environment_metallic : hint_range(0.0, 1.0) = 0.0;

    // ---------------------------------------------------------------------------------
//  // ---------------------------------------------------------------------------------
    // VARYINGS FOR TANGENT SPACE IN LIGHT FUNCTION
//  // VARYINGS FOR TANGENT SPACE IN LIGHT FUNCTION
    // ---------------------------------------------------------------------------------
//  // ---------------------------------------------------------------------------------
    varying vec3 custom_tangent;
//  varying vec3 custom_tangent;
    varying vec3 custom_binormal;
//  varying vec3 custom_binormal;

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
        // The light() function still completely overrides the direct lighting calculation with UTIA
//      // The light() function still completely overrides the direct lighting calculation with UTIA
        ALBEDO = base_color_albedo;
//      ALBEDO = base_color_albedo;
        METALLIC = environment_metallic;
//      METALLIC = environment_metallic;
        ROUGHNESS = environment_roughness;
//      ROUGHNESS = environment_roughness;
        
        // Pass the mesh tangent space to the light function
//      // Pass the mesh tangent space to the light function
        custom_tangent = TANGENT;
//      custom_tangent = TANGENT;
        custom_binormal = BINORMAL;
//      custom_binormal = BINORMAL;
    }
//  }

    // ---------------------------------------------------------------------------------
//  // ---------------------------------------------------------------------------------
    // CUSTOM LIGHTING FUNCTION (UTIA BRDF PG14)
//  // CUSTOM LIGHTING FUNCTION (UTIA BRDF PG14)
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
        
        // Transform Light and View vectors into local tangent space
//      // Transform Light and View vectors into local tangent space
        vec3 local_L = vec3(dot(light_direction, custom_tangent), dot(light_direction, custom_binormal), dot(light_direction, surface_normal));
//      vec3 local_L = vec3(dot(light_direction, custom_tangent), dot(light_direction, custom_binormal), dot(light_direction, surface_normal));
        vec3 local_V = vec3(dot(view_direction, custom_tangent), dot(view_direction, custom_binormal), dot(view_direction, surface_normal));
//      vec3 local_V = vec3(dot(view_direction, custom_tangent), dot(view_direction, custom_binormal), dot(view_direction, surface_normal));
        
        // 1. Calculate Elevation (Theta)
//      // 1. Calculate Elevation (Theta)
        // Angle between the normal (Z in local space) and the vector
//      // Angle between the normal (Z in local space) and the vector
        float theta_L = acos(clamp(local_L.z, 0.0, 1.0));
//      float theta_L = acos(clamp(local_L.z, 0.0, 1.0));
        float theta_V = acos(clamp(local_V.z, 0.0, 1.0));
//      float theta_V = acos(clamp(local_V.z, 0.0, 1.0));
        
        // 2. Calculate Azimuth (Phi)
//      // 2. Calculate Azimuth (Phi)
        float phi_L = atan(local_L.y, local_L.x);
//      float phi_L = atan(local_L.y, local_L.x);
        if (phi_L < 0.0) {
//      if (phi_L < 0.0) {
            phi_L += 2.0 * PI;
//          phi_L += 2.0 * PI;
        }
//      }
        
        float phi_V = atan(local_V.y, local_V.x);
//      float phi_V = atan(local_V.y, local_V.x);
        if (phi_V < 0.0) {
//      if (phi_V < 0.0) {
            phi_V += 2.0 * PI;
//          phi_V += 2.0 * PI;
        }
//      }
        
        // 3. Map to UTIA 288x288 Indices
//      // 3. Map to UTIA 288x288 Indices
        // Elevation steps: 15 degrees = PI / 12 radians
//      // Elevation steps: 15 degrees = PI / 12 radians
        float theta_step = PI / 12.0;
//      float theta_step = PI / 12.0;
        // Azimuth steps: 7.5 degrees = PI / 24 radians
//      // Azimuth steps: 7.5 degrees = PI / 24 radians
        float phi_step = PI / 24.0;
//      float phi_step = PI / 24.0;
        
        // Elevation index: 0 to 5 (6 levels)
//      // Elevation index: 0 to 5 (6 levels)
        float theta_idx_L = clamp(round(theta_L / theta_step), 0.0, 5.0);
//      float theta_idx_L = clamp(round(theta_L / theta_step), 0.0, 5.0);
        float theta_idx_V = clamp(round(theta_V / theta_step), 0.0, 5.0);
//      float theta_idx_V = clamp(round(theta_V / theta_step), 0.0, 5.0);
        
        // Azimuth index: 0 to 47 (48 levels)
//      // Azimuth index: 0 to 47 (48 levels)
        float phi_idx_L = mod(round(phi_L / phi_step), 48.0);
//      float phi_idx_L = mod(round(phi_L / phi_step), 48.0);
        float phi_idx_V = mod(round(phi_V / phi_step), 48.0);
//      float phi_idx_V = mod(round(phi_V / phi_step), 48.0);
        
        // Combine to linear index (0 to 287)
//      // Combine to linear index (0 to 287)
        float idx_L = theta_idx_L * 48.0 + phi_idx_L;
//      float idx_L = theta_idx_L * 48.0 + phi_idx_L;
        float idx_V = theta_idx_V * 48.0 + phi_idx_V;
//      float idx_V = theta_idx_V * 48.0 + phi_idx_V;
        
        // 4. Map to 2D Texture Coordinates (U, V)
//      // 4. Map to 2D Texture Coordinates (U, V)
        // Depending on EXR layout, usually rows=Illumination(L), cols=Viewing(V)
//      // Depending on EXR layout, usually rows=Illumination(L), cols=Viewing(V)
        // Using center of pixel (+ 0.5) to avoid sampling between pixels
//      // Using center of pixel (+ 0.5) to avoid sampling between pixels
        float texture_coord_u = (idx_V + 0.5) / 288.0;
//      float texture_coord_u = (idx_V + 0.5) / 288.0;
        float texture_coord_v = (idx_L + 0.5) / 288.0;
//      float texture_coord_v = (idx_L + 0.5) / 288.0;
        
        vec2 texture_coordinates_2d = vec2(clamp(texture_coord_u, 0.0, 1.0), clamp(texture_coord_v, 0.0, 1.0));
//      vec2 texture_coordinates_2d = vec2(clamp(texture_coord_u, 0.0, 1.0), clamp(texture_coord_v, 0.0, 1.0));
        
        // Sample the UTIA 2D Texture
//      // Sample the UTIA 2D Texture
        vec3 sampled_brdf_color = texture(utia_brdf_texture, texture_coordinates_2d).rgb;
//      vec3 sampled_brdf_color = texture(utia_brdf_texture, texture_coordinates_2d).rgb;
        
        // The Rendering Equation: Radiance = BRDF * Light_Color * Attenuation * NdotL
//      // The Rendering Equation: Radiance = BRDF * Light_Color * Attenuation * NdotL
        float normal_dot_light = max(dot(surface_normal, light_direction), 0.0);
//      float normal_dot_light = max(dot(surface_normal, light_direction), 0.0);
        
        // We assign it to SPECULAR_LIGHT to override standard PBR handling.
//      // We assign it to SPECULAR_LIGHT to override standard PBR handling.
        SPECULAR_LIGHT += sampled_brdf_color * base_color_albedo * LIGHT_COLOR * ATTENUATION * normal_dot_light * exposure_multiplier;
//      SPECULAR_LIGHT += sampled_brdf_color * base_color_albedo * LIGHT_COLOR * ATTENUATION * normal_dot_light * exposure_multiplier;
    }
//  }
