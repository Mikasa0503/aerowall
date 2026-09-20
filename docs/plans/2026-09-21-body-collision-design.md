# Explicit body-collision validation

The pinned air.usd enables bat/collisions but disables base_link/collisions and all four rotor collision shapes. The original-asset geometry trial reports real top/side/bottom bat contacts in all 16 environments but no rotor contacts. Therefore actor-path classification alone cannot detect physical non-bat hits in the original baseline.

Keep the author asset and sustained SingleJuggle baseline unchanged for reference/action analysis. Add a project-only optional override during the initial scene parse to enable the existing drone collision shapes; retain their geometry, articulation, controller and solver. The option defaults off and is currently restricted to the dedicated geometry validation probe. Record the composed collision flags and test top/side/bottom/rotor fixtures. This is an explicit physics configuration change, not a claim of unmodified upstream reproduction.

The future WallRally task must opt into physical body contacts and terminate on non-bat ball collisions and drone/environment collisions. Revalidate reset/isolation/stability in that task before formal training. Do not infer those later gates from this centered fixture, and do not silently change the current baseline while it trains.
