// Delade typer för hela Pingis Collector-appen

export interface ImuSample {
  accel_x: number;
  accel_y: number;
  accel_z: number;
  gyro_x: number;
  gyro_y: number;
  gyro_z: number;
  mag_x: number;
  mag_y: number;
  mag_z: number;
  ts_ms: number;
}

export interface PlayerSetup {
  name: string;
  handedness: 'right' | 'left';
}

export interface CalibrationData {
  gravity: { x: number; y: number; z: number };
  gyro_bias: { x: number; y: number; z: number };
}

export interface LabeledEvent {
  label: 'hit' | 'swing_miss' | 'idle';
  stroke_type: 'forehand' | 'backhand' | 'unknown';
  recorded_at: string;
  samples: ImuSample[];
}

export interface SessionFile {
  session_meta: {
    player_name: string;
    handedness: 'right' | 'left';
    calibration_accel: { x: number; y: number; z: number };
    calibration_gyro_bias: { x: number; y: number; z: number };
    session_date: string;
    app_version: string;
  };
  events: LabeledEvent[];
}
