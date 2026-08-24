# Structured understanding and duration live validation

- Timestamp: `2026-08-24T04:24:50Z`
- Baseline: `4ef9e16`
- Branch: `codex/structured-understanding-duration-fix`
- Credential values: not recorded
- Real video submissions: `0`

## Provider request ledger

| Request | Route/model | Result | Evidence |
|---|---|---|---|
| Event understanding | Agent Plan Chat / `doubao-seed-evolving` | Passed | One native `EventUnderstandingBatch`; source SHA-256 `19ab073d7d20ea1d6dfadbfa4f55814eb508b226569c408516feb1e3249173d4` |
| Visual probe before route fix | Standard PAYG Responses | Rejected before inference | `AuthenticationError` confirmed Agent Plan credential/standard route mismatch |
| Visual controlled route probe | Agent Plan Responses / `doubao-seed-2-0-lite-260428` | Passed | Native `ShotSemanticReview`, verdict `pass`, confidence `0.95` |
| Visual default-client verification | Agent Plan Responses / `doubao-seed-2-0-lite-260428` | Passed | Native `ShotSemanticReview`, verdict `pass`, confidence `0.98` |
| Character probe before prompt reconciliation | Agent Plan Chat / `doubao-seed-evolving` | Valid empty DTO, then fail closed | Proved the model prompt still contradicted source-declared codename policy |
| Character default-client verification | Agent Plan Chat / `doubao-seed-evolving` | Passed | One canonical character `observer`; event bound to `character_ids=["observer"]` and `honcut.semantic-understanding.v1` |

Potentially billable completed inference calls: `5`. Authentication-rejected
requests: `1`. A separate local prefilter failure made `0` Provider requests.
The same existing image, SHA-256
`9f367425517ef76ae023b27dabafd66bb4f661b9fbdec47d04d43aa6f6c65ba5`,
was uploaded to the configured TOS content key for the three visual attempts.

## Duration contract evidence

The regression contract proves that a `3s` effective story beat is carried by
an `8s` Provider request with `5s` Provider padding. The padding is cost and
normalization context; it does not consume the delivery story clock. Material
accounting is persisted as `honcut.material-budget.v3` and fails closed for
stale or incomplete request ledgers.

## Verification

- Full suite before the live route/codename fixes: `975 passed`.
- Targeted post-fix route, schema, cache-envelope, and identity tests passed.
- Final post-fix full suite: `976 passed in 32.88s`.
