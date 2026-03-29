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
    // make sure the Filter is set to "Nearest" and Repeat is "Disabled"
//  // make sure the Filter is set to "Nearest" and Repeat is "Disabled"
    uniform sampler2D utia_brdf_texture : source_color, filter_nearest, repeat_disable;
//  uniform sampler2D utia_brdf_texture : source_color, filter_nearest, repeat_disable;

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

        // We need to match lookup_brdf_val logic from C++ TBRDF.cpp
//      // We need to match lookup_brdf_val logic from C++ TBRDF.cpp
        float theta_i = theta_L;
//      float theta_i = theta_L;
        float theta_v = theta_V;
//      float theta_v = theta_V;
        float phi_i = phi_L;
//      float phi_i = phi_L;
        float phi_v = phi_V;
//      float phi_v = phi_V;

        // Elevation steps: 15 degrees = PI / 12 radians
//      // Elevation steps: 15 degrees = PI / 12 radians
        float theta_step = PI / 12.0;
//      float theta_step = PI / 12.0;
        // Azimuth steps: 7.5 degrees = PI / 24 radians
//      // Azimuth steps: 7.5 degrees = PI / 24 radians
        float phi_step = PI / 24.0;
//      float phi_step = PI / 24.0;

        float f_iti = theta_i / theta_step;
//      float f_iti = theta_i / theta_step;
        float iti0 = floor(f_iti);
//      float iti0 = floor(f_iti);
        float iti1 = iti0 + 1.0;
//      float iti1 = iti0 + 1.0;
        if (iti0 > 4.0) {
//      if (iti0 > 4.0) {
            iti0 = 4.0;
//          iti0 = 4.0;
            iti1 = 5.0;
//          iti1 = 5.0;
        }
//      }
        float wti1 = (theta_i - (theta_step * iti0)) / theta_step;
//      float wti1 = (theta_i - (theta_step * iti0)) / theta_step;
        float wti0 = 1.0 - wti1;
//      float wti0 = 1.0 - wti1;

        float f_itv = theta_v / theta_step;
//      float f_itv = theta_v / theta_step;
        float itv0 = floor(f_itv);
//      float itv0 = floor(f_itv);
        float itv1 = itv0 + 1.0;
//      float itv1 = itv0 + 1.0;
        if (itv0 > 4.0) {
//      if (itv0 > 4.0) {
            itv0 = 4.0;
//          itv0 = 4.0;
            itv1 = 5.0;
//          itv1 = 5.0;
        }
//      }
        float wtv1 = (theta_v - (theta_step * itv0)) / theta_step;
//      float wtv1 = (theta_v - (theta_step * itv0)) / theta_step;
        float wtv0 = 1.0 - wtv1;
//      float wtv0 = 1.0 - wtv1;

        float f_ipi = phi_i / phi_step;
//      float f_ipi = phi_i / phi_step;
        float ipi0 = floor(f_ipi);
//      float ipi0 = floor(f_ipi);
        if (ipi0 >= 48.0) {
//      if (ipi0 >= 48.0) {
            ipi0 = 0.0;
//          ipi0 = 0.0;
        }
//      }
        float ipi1 = ipi0 + 1.0;
//      float ipi1 = ipi0 + 1.0;
        float wpi1 = fract(f_ipi);
//      float wpi1 = fract(f_ipi);
        float wpi0 = 1.0 - wpi1;
//      float wpi0 = 1.0 - wpi1;
        if (ipi1 >= 48.0) {
//      if (ipi1 >= 48.0) {
            ipi1 = 0.0;
//          ipi1 = 0.0;
        }
//      }

        float f_ipv = phi_v / phi_step;
//      float f_ipv = phi_v / phi_step;
        float ipv0 = floor(f_ipv);
//      float ipv0 = floor(f_ipv);
        if (ipv0 >= 48.0) {
//      if (ipv0 >= 48.0) {
            ipv0 = 0.0;
//          ipv0 = 0.0;
        }
//      }
        float ipv1 = ipv0 + 1.0;
//      float ipv1 = ipv0 + 1.0;
        float wpv1 = fract(f_ipv);
//      float wpv1 = fract(f_ipv);
        float wpv0 = 1.0 - wpv1;
//      float wpv0 = 1.0 - wpv1;
        if (ipv1 >= 48.0) {
//      if (ipv1 >= 48.0) {
            ipv1 = 0.0;
//          ipv1 = 0.0;
        }
//      }

        float wti[2];
//      float wti[2];
        wti[0] = wti0; wti[1] = wti1;
//      wti[0] = wti0; wti[1] = wti1;
        float wtv[2];
//      float wtv[2];
        wtv[0] = wtv0; wtv[1] = wtv1;
//      wtv[0] = wtv0; wtv[1] = wtv1;
        float wpi[2];
//      float wpi[2];
        wpi[0] = wpi0; wpi[1] = wpi1;
//      wpi[0] = wpi0; wpi[1] = wpi1;
        float wpv[2];
//      float wpv[2];
        wpv[0] = wpv0; wpv[1] = wpv1;
//      wpv[0] = wpv0; wpv[1] = wpv1;

        float iti[2];
//      float iti[2];
        iti[0] = iti0; iti[1] = iti1;
//      iti[0] = iti0; iti[1] = iti1;
        float itv[2];
//      float itv[2];
        itv[0] = itv0; itv[1] = itv1;
//      itv[0] = itv0; itv[1] = itv1;
        float ipi[2];
//      float ipi[2];
        ipi[0] = ipi0; ipi[1] = ipi1;
//      ipi[0] = ipi0; ipi[1] = ipi1;
        float ipv[2];
//      float ipv[2];
        ipv[0] = ipv0; ipv[1] = ipv1;
//      ipv[0] = ipv0; ipv[1] = ipv1;

        vec3 sampled_brdf_color = vec3(0.0);
//      vec3 sampled_brdf_color = vec3(0.0);

        for (int i = 0; i < 2; i++) {
//      for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
//          for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
//              for (int k = 0; k < 2; k++) {
                    for (int l = 0; l < 2; l++) {
//                  for (int l = 0; l < 2; l++) {
                        float idx_L = iti[i] * 48.0 + ipi[k];
//                      float idx_L = iti[i] * 48.0 + ipi[k];
                        float idx_V = itv[j] * 48.0 + ipv[l];
//                      float idx_V = itv[j] * 48.0 + ipv[l];

                        // Depending on EXR layout, rows=Illumination(L), cols=Viewing(V)
//                      // Depending on EXR layout, rows=Illumination(L), cols=Viewing(V)
                        // Using center of pixel (+ 0.5) to avoid sampling between pixels
//                      // Using center of pixel (+ 0.5) to avoid sampling between pixels
                        float texture_coord_u = (idx_V + 0.5) / 288.0;
//                      float texture_coord_u = (idx_V + 0.5) / 288.0;
                        float texture_coord_v = (idx_L + 0.5) / 288.0;
//                      float texture_coord_v = (idx_L + 0.5) / 288.0;

                        vec2 texture_coordinates_2d = vec2(clamp(texture_coord_u, 0.0, 1.0), clamp(texture_coord_v, 0.0, 1.0));
//                      vec2 texture_coordinates_2d = vec2(clamp(texture_coord_u, 0.0, 1.0), clamp(texture_coord_v, 0.0, 1.0));

                        vec3 tex_val = textureLod(utia_brdf_texture, texture_coordinates_2d, 0.0).rgb;
//                      vec3 tex_val = textureLod(utia_brdf_texture, texture_coordinates_2d, 0.0).rgb;
                        sampled_brdf_color += tex_val * wti[i] * wtv[j] * wpi[k] * wpv[l];
//                      sampled_brdf_color += tex_val * wti[i] * wtv[j] * wpi[k] * wpv[l];
                    }
//                  }
                }
//              }
            }
//          }
        }
//      }

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
