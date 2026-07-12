import type { CSSProperties } from "react";
import type {
  NotebookDraft,
  NotebookProfileMetadata,
  NotebookQualityProfileId,
} from "./types";

interface Props {
  profiles: readonly NotebookProfileMetadata[];
  value: NotebookDraft["quality"];
  errors: Partial<Record<"profile" | "nIters" | "maxGaussians", string>>;
  onProfileChange(profile: NotebookQualityProfileId): void;
  onIterationsChange(value: string): void;
  onMaxGaussiansChange(value: string): void;
  onResetOverrides(): void;
}

export function QualitySection(props: Props) {
  const maxBudget = Math.max(...props.profiles.map((profile) => profile.max_gaussians));
  const active = props.profiles.find((profile) => profile.id === props.value.profile);
  return (
    <>
      <section className="nb-stage is-complete">
        <div className="nb-stage-dot" aria-hidden="true" />
        <div className="nb-stage-body">
          <p className="nb-stage-kicker">03 / compute</p>
          <h2>Quality profile</h2>
          <div className="nb-vram-ladder">
            {props.profiles.map((profile) => {
              const budget = Math.max(12, (profile.max_gaussians / maxBudget) * 100);
              return (
                <label
                  key={profile.id}
                  className={`nb-profile ${profile.id === props.value.profile ? "is-selected" : ""}`}
                  style={{ "--nb-budget": `${budget}%` } as CSSProperties}
                >
                  <input
                    type="radio"
                    name="quality-profile"
                    checked={profile.id === props.value.profile}
                    onChange={() => props.onProfileChange(profile.id)}
                  />
                  <span className="nb-profile-copy">
                    <strong>{profile.label}</strong>
                    <small>{profile.minimum_vram_gb}+ GB · {profile.selected_frame_budget} frames</small>
                  </span>
                  <span className="nb-profile-budget" aria-hidden="true"><i /></span>
                  <span className="nb-profile-gauss">{(profile.max_gaussians / 1_000_000).toFixed(2)}M</span>
                </label>
              );
            })}
          </div>
          {props.errors.profile && <p className="nb-error">{props.errors.profile}</p>}
        </div>
      </section>
      <section className="nb-stage is-complete">
        <div className="nb-stage-dot" aria-hidden="true" />
        <div className="nb-stage-body">
          <p className="nb-stage-kicker">04 / limits</p>
          <h2>Iterations and maximum Gaussians</h2>
          <div className="nb-inline-fields">
            <label className="nb-field">
              <span>Iterations <em>default {active?.n_iters.toLocaleString()}</em></span>
              <input
                type="number"
                value={props.value.nIters}
                placeholder="Profile default"
                onChange={(event) => props.onIterationsChange(event.target.value)}
                aria-invalid={Boolean(props.errors.nIters)}
              />
              {props.errors.nIters && <small className="nb-error">{props.errors.nIters}</small>}
            </label>
            <label className="nb-field">
              <span>Maximum Gaussians <em>default {active?.max_gaussians.toLocaleString()}</em></span>
              <input
                type="number"
                value={props.value.maxGaussians}
                placeholder="Profile default"
                onChange={(event) => props.onMaxGaussiansChange(event.target.value)}
                aria-invalid={Boolean(props.errors.maxGaussians)}
              />
              {props.errors.maxGaussians && <small className="nb-error">{props.errors.maxGaussians}</small>}
            </label>
          </div>
          <button type="button" className="nb-text-button" onClick={props.onResetOverrides}>
            Reset primary overrides
          </button>
        </div>
      </section>
    </>
  );
}
