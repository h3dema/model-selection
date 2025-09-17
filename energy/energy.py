import math
import random



def solar_power_estimate(
    n, t_solar, latitude, tilt, azimuth,
    area=1.6,
    eta_ref=0.18,
    alpha_T=-0.004,
    NOCT=45,
    T_amb=25,
    rho_g=0.2,
    k=0.14
):
    G_sc = 1367  # solar constant

    # declination
    delta = math.radians(23.45 * math.sin(math.radians(360 * (284 + n) / 365)))

    # latitude etc.
    phi = math.radians(latitude)
    beta = math.radians(tilt)
    gamma_p = math.radians(azimuth)
    omega = math.radians(15 * (t_solar - 12))

    # zenith angle
    cos_theta_z = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(omega)
    if cos_theta_z <= 0:
        return 0.0

    # incidence angle
    cos_theta = (
        math.sin(delta) * math.sin(phi) * math.cos(beta)
        - math.sin(delta) * math.cos(phi) * math.sin(beta) * math.cos(gamma_p)
        + math.cos(delta) * math.cos(phi) * math.cos(beta) * math.cos(omega)
        + math.cos(delta) * math.sin(phi) * math.sin(beta) * math.cos(gamma_p) * math.cos(omega)
        + math.cos(delta) * math.sin(beta) * math.sin(gamma_p) * math.sin(omega)
    )
    cos_theta = max(0, cos_theta)

    # extraterrestrial irradiance
    E0 = 1 + 0.033 * math.cos(math.radians(360 * n / 365))
    G0 = G_sc * E0

    # clear-sky attenuation
    AM = 1 / cos_theta_z
    G_b = G0 * math.exp(-k * AM)
    G_b_horizontal = G_b * cos_theta_z

    # diffuse assumption
    G_d = 0.15 * (G_b_horizontal / cos_theta_z)
    G_h = G_b_horizontal + G_d

    # tilt adjustments
    G_b_tilt = G_b * cos_theta
    G_d_tilt = G_d * (1 + math.cos(beta)) / 2
    G_r = G_h * rho_g * (1 - math.cos(beta)) / 2
    G_POA = G_b_tilt + G_d_tilt + G_r

    # cell temp & efficiency
    T_cell = T_amb + (NOCT - 20) / 800 * G_POA
    eta = eta_ref * (1 + alpha_T * (T_cell - 25))
    return max(G_POA * area * eta, 0.0)


def day_weather_state():
    r = random.random()
    if r < 0.6:
        return "clear"
    elif r < 0.85:
        return "partly"
    else:
        return "cloudy"


# weather-state model → best for yearly/weekly studies with “realistic” cloudy vs sunny distributions.
def weather_factor(state):
    if state == "clear": return random.uniform(0.9, 1.0)
    if state == "partly": return random.uniform(0.4, 0.8)
    if state == "cloudy": return random.uniform(0.1, 0.3)


def solar_day_curve(n, hours, latitude, tilt, azimuth, **kwargs):
    return [solar_power_estimate(n, h, latitude, tilt, azimuth, **kwargs) for h in hours]


def solar_day_curve_with_weather(n, hours, latitude, tilt, azimuth, area=1.6, **kwargs):
    base_curve = [solar_power_estimate(n, h, latitude, tilt, azimuth, area=area, **kwargs) for h in hours]

    # assign a weather state for this day
    state = day_weather_state()
    factor = weather_factor(state)

    return [p * factor for p in base_curve], state
