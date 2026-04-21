    shader_type spatial;
//  shader_type spatial;
    render_mode blend_mix, depth_draw_always, cull_disabled, diffuse_burley, specular_schlick_ggx;
//  render_mode blend_mix, depth_draw_always, cull_disabled, diffuse_burley, specular_schlick_ggx;

    // --- Baked Glass Look-Up Tables (LUTs) ---
//  // --- Baked Glass Look-Up Tables (LUTs) ---
    // The 2D Look-Up Table (LUT) for Multiple Scattering Energy Compensation.
//  // The 2D Look-Up Table (LUT) for Multiple Scattering Energy Compensation.
    // This texture maps the relationship between the viewing angle (N.V) and surface roughness to recover lost energy.
//  // This texture maps the relationship between the viewing angle (N.V) and surface roughness to recover lost energy.
    // R = Reflected Energy (Er), G = Transmitted Energy (Et), B = Total Energy (Er + Et).
//  // R = Reflected Energy (Er), G = Transmitted Energy (Et), B = Total Energy (Er + Et).
    uniform sampler2D glass_lut_energy_comp_texture : hint_default_black, filter_linear, repeat_disable;
//  uniform sampler2D glass_lut_energy_comp_texture : hint_default_black, filter_linear, repeat_disable;

    // The 2D Look-Up Table (LUT) for Pre-integrated Rough Transmission.
//  // The 2D Look-Up Table (LUT) for Pre-integrated Rough Transmission.
    // This texture maps the relationship between the viewing angle (N.V) and surface roughness to simulate blurred refraction.
//  // This texture maps the relationship between the viewing angle (N.V) and surface roughness to simulate blurred refraction.
    // R = Split-Sum Base (Et), G = Split-Sum Bias, B = Roughness mapping for screen texture mipmap selection.
//  // R = Split-Sum Base (Et), G = Split-Sum Bias, B = Roughness mapping for screen texture mipmap selection.
    uniform sampler2D glass_lut_transmission_texture : hint_default_black, filter_linear, repeat_disable;
//  uniform sampler2D glass_lut_transmission_texture : hint_default_black, filter_linear, repeat_disable;

    // The 1D Look-Up Table (LUT) for Volumetric Absorption based on the Beer-Lambert Law.
//  // The 1D Look-Up Table (LUT) for Volumetric Absorption based on the Beer-Lambert Law.
    // This texture maps the simulated thickness of the glass to the color transmittance profile.
//  // This texture maps the simulated thickness of the glass to the color transmittance profile.
    // It is parameterized by a maximum thickness value of 10.0 units, defined during the baking process.
//  // It is parameterized by a maximum thickness value of 10.0 units, defined during the baking process.
    uniform sampler2D glass_lut_absorption_texture : hint_default_white, filter_linear, repeat_disable;
//  uniform sampler2D glass_lut_absorption_texture : hint_default_white, filter_linear, repeat_disable;

    // --- Built-in Screen Texture for Refraction Sampling ---
//  // --- Built-in Screen Texture for Refraction Sampling ---
    // This texture captures the opaque scene rendered behind the glass object to be distorted for refraction.
//  // This texture captures the opaque scene rendered behind the glass object to be distorted for refraction.
    uniform sampler2D screen_texture : hint_screen_texture, filter_linear_mipmap, repeat_disable;
//  uniform sampler2D screen_texture : hint_screen_texture, filter_linear_mipmap, repeat_disable;

    // --- Glass Material Core Properties ---
//  // --- Glass Material Core Properties ---
    // The foundational transmission color or tint of the glass material before volumetric absorption is applied.
//  // The foundational transmission color or tint of the glass material before volumetric absorption is applied.
    uniform vec3 glass_base_color : source_color = vec3(1.0, 1.0, 1.0);
//  uniform vec3 glass_base_color : source_color = vec3(1.0, 1.0, 1.0);
    // The perceived microscopic roughness of the glass surface, blurring both reflections and refractions.
//  // The perceived microscopic roughness of the glass surface, blurring both reflections and refractions.
    uniform float glass_roughness : hint_range(0.0, 1.0) = 0.1;
//  uniform float glass_roughness : hint_range(0.0, 1.0) = 0.1;
    // The Index of Refraction (IOR), which determines how much light bends when entering the glass medium.
//  // The Index of Refraction (IOR), which determines how much light bends when entering the glass medium.
    uniform float glass_ior : hint_range(1.0, 2.5) = 1.5;
//  uniform float glass_ior : hint_range(1.0, 2.5) = 1.5;

    // Controls the simulated geometric depth of the glass object for volumetric light absorption calculations.
//  // Controls the simulated geometric depth of the glass object for volumetric light absorption calculations.
    uniform float glass_thickness : hint_range(0.0, 10.0) = 1.0;
//  uniform float glass_thickness : hint_range(0.0, 10.0) = 1.0;

    // Controls the intensity of the chromatic aberration effect, simulating wavelength-dependent light dispersion.
//  // Controls the intensity of the chromatic aberration effect, simulating wavelength-dependent light dispersion.
    uniform float chromatic_aberration_intensity : hint_range(0.0, 1.0) = 0.05;
//  uniform float chromatic_aberration_intensity : hint_range(0.0, 1.0) = 0.05;

    // A multiplier for the intensity of the recovered multiple scattering energy, allowing artistic adjustments.
//  // A multiplier for the intensity of the recovered multiple scattering energy, allowing artistic adjustments.
    uniform float energy_compensation_multiplier : hint_range(0.0, 5.0) = 1.0;
//  uniform float energy_compensation_multiplier : hint_range(0.0, 5.0) = 1.0;

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
        // 1. Core Vectors and Viewing Angle Calculation
//      // 1. Core Vectors and Viewing Angle Calculation
        // Normalize the core vectors required for sampling look-up tables and refraction calculations.
//      // Normalize the core vectors required for sampling look-up tables and refraction calculations.
        vec3 normalized_surface_normal_vector = normalize(NORMAL);
//      vec3 normalized_surface_normal_vector = normalize(NORMAL);
        vec3 normalized_outgoing_view_direction = normalize(VIEW);
//      vec3 normalized_outgoing_view_direction = normalize(VIEW);
        // Calculate the cosine of the angle between the surface normal and the view direction, ensuring it is non-negative.
//      // Calculate the cosine of the angle between the surface normal and the view direction, ensuring it is non-negative.
        float dot_product_normal_and_view_direction = max(0.0, dot(normalized_surface_normal_vector, normalized_outgoing_view_direction));
//      float dot_product_normal_and_view_direction = max(0.0, dot(normalized_surface_normal_vector, normalized_outgoing_view_direction));

        // 2. Sample Pre-calculated Look-Up Tables
//      // 2. Sample Pre-calculated Look-Up Tables
        // Construct the 2D UV coordinates based on the viewing angle and the surface roughness parameter.
//      // Construct the 2D UV coordinates based on the viewing angle and the surface roughness parameter.
        vec2 calculated_lut_sampling_uv_coordinates = vec2(dot_product_normal_and_view_direction, glass_roughness);
//      vec2 calculated_lut_sampling_uv_coordinates = vec2(dot_product_normal_and_view_direction, glass_roughness);
        vec3 sampled_energy_compensation_color = texture(glass_lut_energy_comp_texture, calculated_lut_sampling_uv_coordinates).rgb;
//      vec3 sampled_energy_compensation_color = texture(glass_lut_energy_comp_texture, calculated_lut_sampling_uv_coordinates).rgb;
        vec3 sampled_transmission_lut_color = texture(glass_lut_transmission_texture, calculated_lut_sampling_uv_coordinates).rgb;
//      vec3 sampled_transmission_lut_color = texture(glass_lut_transmission_texture, calculated_lut_sampling_uv_coordinates).rgb;

        // The Volumetric Absorption Look-Up Table is 1-Dimensional and parameterized purely by object thickness.
//      // The Volumetric Absorption Look-Up Table is 1-Dimensional and parameterized purely by object thickness.
        // We normalize the user-defined thickness against the maximum baked limit of 10.0 to find the correct texture coordinate.
//      // We normalize the user-defined thickness against the maximum baked limit of 10.0 to find the correct texture coordinate.
        float calculated_volumetric_thickness_u_coordinate = clamp(glass_thickness / 10.0, 0.0, 1.0);
//      float calculated_volumetric_thickness_u_coordinate = clamp(glass_thickness / 10.0, 0.0, 1.0);
        vec3 sampled_volumetric_absorption_tint_color = texture(glass_lut_absorption_texture, vec2(calculated_volumetric_thickness_u_coordinate, 0.5)).rgb;
//      vec3 sampled_volumetric_absorption_tint_color = texture(glass_lut_absorption_texture, vec2(calculated_volumetric_thickness_u_coordinate, 0.5)).rgb;

        // 3. Compute Refraction Direction Vector via Snell's Law
//      // 3. Compute Refraction Direction Vector via Snell's Law
        // Calculate the relative index of refraction, assuming light is transitioning from standard air (IOR 1.0) into the glass medium.
//      // Calculate the relative index of refraction, assuming light is transitioning from standard air (IOR 1.0) into the glass medium.
        float calculated_relative_index_of_refraction_eta = 1.0 / glass_ior;
//      float calculated_relative_index_of_refraction_eta = 1.0 / glass_ior;
        vec3 calculated_refraction_direction_vector = refract(-normalized_outgoing_view_direction, normalized_surface_normal_vector, calculated_relative_index_of_refraction_eta);
//      vec3 calculated_refraction_direction_vector = refract(-normalized_outgoing_view_direction, normalized_surface_normal_vector, calculated_relative_index_of_refraction_eta);

        // Project the computed 3D refraction direction into 2D screen UV space.
//      // Project the computed 3D refraction direction into 2D screen UV space.
        // This offset is linearly approximated based on the simulated object thickness to distort the background realistically.
//      // This offset is linearly approximated based on the simulated object thickness to distort the background realistically.
        vec2 calculated_screen_space_refraction_uv_offset = calculated_refraction_direction_vector.xy * (glass_thickness * 0.05);
//      vec2 calculated_screen_space_refraction_uv_offset = calculated_refraction_direction_vector.xy * (glass_thickness * 0.05);

        // 4. Determine Screen Texture Mipmap Level for Rough Transmission Simulation
//      // 4. Determine Screen Texture Mipmap Level for Rough Transmission Simulation
        // The blue channel of the transmission LUT contains the pre-calculated roughness mapping coefficient.
//      // The blue channel of the transmission LUT contains the pre-calculated roughness mapping coefficient.
        // We scale this mapped roughness by the maximum expected mipmap lod levels to determine the sampling blurriness.
//      // We scale this mapped roughness by the maximum expected mipmap lod levels to determine the sampling blurriness.
        float maximum_mipmap_level_of_detail = 7.0;
//      float maximum_mipmap_level_of_detail = 7.0;
        float calculated_roughness_mipmap_level_of_detail = sampled_transmission_lut_color.b * maximum_mipmap_level_of_detail;
//      float calculated_roughness_mipmap_level_of_detail = sampled_transmission_lut_color.b * maximum_mipmap_level_of_detail;

        // 5. Sample the Screen Texture and Apply Simulated Chromatic Aberration Dispersion
//      // 5. Sample the Screen Texture and Apply Simulated Chromatic Aberration Dispersion
        vec3 sampled_refracted_screen_texture_color = vec3(0.0);
//      vec3 sampled_refracted_screen_texture_color = vec3(0.0);
        if (chromatic_aberration_intensity > 0.001) {
//      if (chromatic_aberration_intensity > 0.001) {
            // We sample the red, green, and blue color channels independently with slight positional UV offsets.
//          // We sample the red, green, and blue color channels independently with slight positional UV offsets.
            // This multi-sampling simulates the physical dispersion of varying light wavelengths passing through the dense medium.
//          // This multi-sampling simulates the physical dispersion of varying light wavelengths passing through the dense medium.
            float calculated_chromatic_aberration_uv_offset = chromatic_aberration_intensity * 0.02 * glass_thickness;
//          float calculated_chromatic_aberration_uv_offset = chromatic_aberration_intensity * 0.02 * glass_thickness;
            sampled_refracted_screen_texture_color.r = textureLod(screen_texture, SCREEN_UV + calculated_screen_space_refraction_uv_offset * (1.0 + calculated_chromatic_aberration_uv_offset), calculated_roughness_mipmap_level_of_detail).r;
//          sampled_refracted_screen_texture_color.r = textureLod(screen_texture, SCREEN_UV + calculated_screen_space_refraction_uv_offset * (1.0 + calculated_chromatic_aberration_uv_offset), calculated_roughness_mipmap_level_of_detail).r;
            sampled_refracted_screen_texture_color.g = textureLod(screen_texture, SCREEN_UV + calculated_screen_space_refraction_uv_offset, calculated_roughness_mipmap_level_of_detail).g;
//          sampled_refracted_screen_texture_color.g = textureLod(screen_texture, SCREEN_UV + calculated_screen_space_refraction_uv_offset, calculated_roughness_mipmap_level_of_detail).g;
            sampled_refracted_screen_texture_color.b = textureLod(screen_texture, SCREEN_UV + calculated_screen_space_refraction_uv_offset * (1.0 - calculated_chromatic_aberration_uv_offset), calculated_roughness_mipmap_level_of_detail).b;
//          sampled_refracted_screen_texture_color.b = textureLod(screen_texture, SCREEN_UV + calculated_screen_space_refraction_uv_offset * (1.0 - calculated_chromatic_aberration_uv_offset), calculated_roughness_mipmap_level_of_detail).b;
        } else {
//      } else {
            // If the chromatic aberration effect is disabled, perform a single unified texture sample for efficiency.
//          // If the chromatic aberration effect is disabled, perform a single unified texture sample for efficiency.
            sampled_refracted_screen_texture_color = textureLod(screen_texture, SCREEN_UV + calculated_screen_space_refraction_uv_offset, calculated_roughness_mipmap_level_of_detail).rgb;
//          sampled_refracted_screen_texture_color = textureLod(screen_texture, SCREEN_UV + calculated_screen_space_refraction_uv_offset, calculated_roughness_mipmap_level_of_detail).rgb;
        }
//      }

        // 6. Combine Core Transmission with Base Material Color and Look-Up Tables
//      // 6. Combine Core Transmission with Base Material Color and Look-Up Tables
        // The final refracted background light is systematically modulated by the foundational base color of the material,
//      // The final refracted background light is systematically modulated by the foundational base color of the material,
        // the calculated volumetric absorption tint profile, and the pre-integrated transmission split-sum base.
//      // the calculated volumetric absorption tint profile, and the pre-integrated transmission split-sum base.
        vec3 calculated_final_transmission_color = sampled_refracted_screen_texture_color * glass_base_color * sampled_volumetric_absorption_tint_color * sampled_transmission_lut_color.r;
//      vec3 calculated_final_transmission_color = sampled_refracted_screen_texture_color * glass_base_color * sampled_volumetric_absorption_tint_color * sampled_transmission_lut_color.r;

        // 7. Apply the Multiple Scattering Energy Compensation Framework
//      // 7. Apply the Multiple Scattering Energy Compensation Framework
        // The green color channel of the energy compensation LUT stores the pre-calculated pre-integrated transmitted energy.
//      // The green color channel of the energy compensation LUT stores the pre-calculated pre-integrated transmitted energy.
        // By additively accumulating this energy back into the transmission pass, physically rough glass materials correctly
//      // By additively accumulating this energy back into the transmission pass, physically rough glass materials correctly
        // maintain their realistic ambient brightness instead of artificially darkening at high microfacet roughness values.
//      // maintain their realistic ambient brightness instead of artificially darkening at high microfacet roughness values.
        calculated_final_transmission_color += calculated_final_transmission_color * sampled_energy_compensation_color.g * energy_compensation_multiplier;
//      calculated_final_transmission_color += calculated_final_transmission_color * sampled_energy_compensation_color.g * energy_compensation_multiplier;

        // 8. Output Final Shading Parameters to the Rendering Pipeline
//      // 8. Output Final Shading Parameters to the Rendering Pipeline
        // Because we manually composite the background layer to simulate a perfectly transmissive dielectric medium,
//      // Because we manually composite the background layer to simulate a perfectly transmissive dielectric medium,
        // the standard physically based rendering (PBR) diffuse ALBEDO property must be explicitly set to absolute black.
//      // the standard physically based rendering (PBR) diffuse ALBEDO property must be explicitly set to absolute black.
        // This strict override prevents the engine's standard ambient diffuse lighting from undesirably washing out the clear glass appearance.
//      // This strict override prevents the engine's standard ambient diffuse lighting from undesirably washing out the clear glass appearance.
        ALBEDO = vec3(0.0);
//      ALBEDO = vec3(0.0);

        // Inject our customly constructed refracted transmission light profile directly into the EMISSION output parameter.
//      // Inject our customly constructed refracted transmission light profile directly into the EMISSION output parameter.
        // This forces the transmitted light to behave structurally as an unlit surface component within the rendering engine's forward pass.
//      // This forces the transmitted light to behave structurally as an unlit surface component within the rendering engine's forward pass.
        EMISSION = calculated_final_transmission_color;
//      EMISSION = calculated_final_transmission_color;

        // Set standard PBR surface properties to dictate how the material interacts with external environmental light reflections.
//      // Set standard PBR surface properties to dictate how the material interacts with external environmental light reflections.
        ROUGHNESS = glass_roughness;
//      ROUGHNESS = glass_roughness;
        METALLIC = 0.0; // Glass is fundamentally a dielectric insulator material, thus the metallic property is strictly fixed to 0.0.
//      METALLIC = 0.0; // Glass is fundamentally a dielectric insulator material, thus the metallic property is strictly fixed to 0.0.

        // Compute the engine-specific SPECULAR shading parameter mathematically derived from the configured Index of Refraction (IOR).
//      // Compute the engine-specific SPECULAR shading parameter mathematically derived from the configured Index of Refraction (IOR).
        // First, calculate the fundamental Fresnel Reflectance at normal incidence (F0) using the standard dielectric equation: F0 = ((IOR - 1)/(IOR + 1))^2.
//      // First, calculate the fundamental Fresnel Reflectance at normal incidence (F0) using the standard dielectric equation: F0 = ((IOR - 1)/(IOR + 1))^2.
        // The Godot engine maps a SPECULAR input of 0.5 to a baseline F0 of 0.04. Therefore, the conversion formula is F0 = 0.08 * SPECULAR.
//      // The Godot engine maps a SPECULAR input of 0.5 to a baseline F0 of 0.04. Therefore, the conversion formula is F0 = 0.08 * SPECULAR.
        float calculated_fresnel_reflectance_at_normal_incidence_f0 = pow((glass_ior - 1.0) / (glass_ior + 1.0), 2.0);
//      float calculated_fresnel_reflectance_at_normal_incidence_f0 = pow((glass_ior - 1.0) / (glass_ior + 1.0), 2.0);
        SPECULAR = calculated_fresnel_reflectance_at_normal_incidence_f0 / 0.08;
//      SPECULAR = calculated_fresnel_reflectance_at_normal_incidence_f0 / 0.08;
    }
//  }

    void light() {
//  void light() {
        // Because realistic glass is a dielectric medium that exclusively reflects or transmits incoming light rays,
//      // Because realistic glass is a dielectric medium that exclusively reflects or transmits incoming light rays,
        // we explicitly disable the Godot engine's default diffuse lambertian scattering evaluation entirely.
//      // we explicitly disable the Godot engine's default diffuse lambertian scattering evaluation entirely.
        // This crucial step is required to maintain physically accurate volumetric glass visual characteristics.
//      // This crucial step is required to maintain physically accurate volumetric glass visual characteristics.
        DIFFUSE_LIGHT = vec3(0.0);
//      DIFFUSE_LIGHT = vec3(0.0);

        // The Godot engine's standard built-in SPECULAR_LIGHT calculation is allowed to process normally,
//      // The Godot engine's standard built-in SPECULAR_LIGHT calculation is allowed to process normally,
        // as it accurately handles the GGX microfacet model required for direct surface specular reflection highlights.
//      // as it accurately handles the GGX microfacet model required for direct surface specular reflection highlights.
    }
//  }
