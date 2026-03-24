    // ---------------------------------------------------------------------------------
//  // ---------------------------------------------------------------------------------
    // MERL BRDF Unity URP Custom Function Node
//  // MERL BRDF Unity URP Custom Function Node
    // For use in URP 14+ Shader Graph
//  // For use in URP 14+ Shader Graph
    // ---------------------------------------------------------------------------------
//  // ---------------------------------------------------------------------------------

    #ifndef MERL_BRDF_INCLUDED
//  #ifndef MERL_BRDF_INCLUDED
    #define MERL_BRDF_INCLUDED
//  #define MERL_BRDF_INCLUDED

    // Computes the UVW coordinates to sample a MERL 3D Texture
//  // Computes the UVW coordinates to sample a MERL 3D Texture
    //
//  //
    // Inputs:
//  // Inputs:
    // light_direction = Light Direction (World Space, Normalized)
//  // light_direction = Light Direction (World Space, Normalized)
    // view_direction = View Direction (World Space, Normalized)
//  // view_direction = View Direction (World Space, Normalized)
    // surface_normal = Surface Normal (World Space, Normalized)
//  // surface_normal = Surface Normal (World Space, Normalized)
    //
//  //
    // Outputs:
//  // Outputs:
    // out_texture_coordinates = Float3 containing the [0..1] texture coordinates for the Sample Texture 3D node
//  // out_texture_coordinates = Float3 containing the [0..1] texture coordinates for the Sample Texture 3D node
    void CalculateMERLTextureCoordinates_float(float3 light_direction, float3 view_direction, float3 surface_normal, out float3 out_texture_coordinates) {
//  void CalculateMERLTextureCoordinates_float(float3 light_direction, float3 view_direction, float3 surface_normal, out float3 out_texture_coordinates) {
        // 1. Calculate Half Vector
//      // 1. Calculate Half Vector
        float3 half_vector = normalize(light_direction + view_direction);
//      float3 half_vector = normalize(light_direction + view_direction);

        // 2. Theta_Half
//      // 2. Theta_Half
        float cosine_angle_half = clamp(dot(surface_normal, half_vector), 0.0, 1.0);
//      float cosine_angle_half = clamp(dot(surface_normal, half_vector), 0.0, 1.0);
        float angle_theta_half = acos(cosine_angle_half);
//      float angle_theta_half = acos(cosine_angle_half);

        // 3. Theta_Diff
//      // 3. Theta_Diff
        float cosine_angle_diff = clamp(dot(light_direction, half_vector), 0.0, 1.0);
//      float cosine_angle_diff = clamp(dot(light_direction, half_vector), 0.0, 1.0);
        float angle_theta_diff = acos(cosine_angle_diff);
//      float angle_theta_diff = acos(cosine_angle_diff);

        // 4. Phi_Diff (Azimuthal difference using Vector Projections)
//      // 4. Phi_Diff (Azimuthal difference using Vector Projections)
        float3 binormal_y_axis = cross(surface_normal, half_vector);
//      float3 binormal_y_axis = cross(surface_normal, half_vector);
        float binormal_length = length(binormal_y_axis);
//      float binormal_length = length(binormal_y_axis);

        if (binormal_length < 1e-5) {
//      if (binormal_length < 1e-5) {
            // Fallback if N and H are completely parallel
//          // Fallback if N and H are completely parallel
            binormal_y_axis = cross(surface_normal, float3(1.0, 0.0, 0.0));
//          binormal_y_axis = cross(surface_normal, float3(1.0, 0.0, 0.0));
            if (length(binormal_y_axis) < 1e-5) {
//          if (length(binormal_y_axis) < 1e-5) {
                binormal_y_axis = cross(surface_normal, float3(0.0, 1.0, 0.0));
//              binormal_y_axis = cross(surface_normal, float3(0.0, 1.0, 0.0));
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

        float3 tangent_x_axis = normalize(cross(binormal_y_axis, half_vector));
//      float3 tangent_x_axis = normalize(cross(binormal_y_axis, half_vector));

        float light_projected_x = dot(light_direction, tangent_x_axis);
//      float light_projected_x = dot(light_direction, tangent_x_axis);
        float light_projected_y = dot(light_direction, binormal_y_axis);
//      float light_projected_y = dot(light_direction, binormal_y_axis);

        float angle_phi_diff = atan2(light_projected_y, light_projected_x);
//      float angle_phi_diff = atan2(light_projected_y, light_projected_x);
        if (angle_phi_diff < 0.0) {
//      if (angle_phi_diff < 0.0) {
            angle_phi_diff += 3.14159265359;
//          angle_phi_diff += 3.14159265359;
        }
//      }

        // 5. Map to 3D Texture Coordinates [0..1]
//      // 5. Map to 3D Texture Coordinates [0..1]
        // Note the non-linear sqrt mapping for Theta_H (W axis)
//      // Note the non-linear sqrt mapping for Theta_H (W axis)
        float texture_depth_w = sqrt(angle_theta_half / (3.14159265359 * 0.5));
//      float texture_depth_w = sqrt(angle_theta_half / (3.14159265359 * 0.5));
        float texture_coord_v = angle_theta_diff / (3.14159265359 * 0.5);
//      float texture_coord_v = angle_theta_diff / (3.14159265359 * 0.5);
        float texture_coord_u = angle_phi_diff / 3.14159265359;
//      float texture_coord_u = angle_phi_diff / 3.14159265359;

        out_texture_coordinates = float3(clamp(texture_coord_u, 0.0, 1.0), clamp(texture_coord_v, 0.0, 1.0), clamp(texture_depth_w, 0.0, 1.0));
//      out_texture_coordinates = float3(clamp(texture_coord_u, 0.0, 1.0), clamp(texture_coord_v, 0.0, 1.0), clamp(texture_depth_w, 0.0, 1.0));
    }
//  }

    #endif
//  #endif
