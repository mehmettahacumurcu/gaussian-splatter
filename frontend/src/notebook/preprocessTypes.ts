export type SettingValue = string | number | boolean;
export interface PreprocessInput { kind: 'video' | 'photos'; path: string; fps: number }
export interface PreprocessSpec {
  schema_version: 1;
  inputs: PreprocessInput[];
  output_path: string;
  dataset_name: string;
  settings: Record<string, SettingValue>;
}
export interface PreprocessField {
  title: string;
  description?: string;
  section: string;
  type: string;
  default: SettingValue;
  enum?: string[];
  when?: string | null;
  minimum?: number;
  maximum?: number;
  maxLength?: number;
}
export interface PreprocessCatalog {
  schema_version: number;
  spirula_version: string;
  groups: string[];
  fields: Record<string, PreprocessField>;
  defaults: Record<string, SettingValue>;
}
