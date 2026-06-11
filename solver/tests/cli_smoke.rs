//! Smoke test for the `solve` profiling CLI. Runs the built binary on a fast
//! board and checks the output line shape and value.

#[test]
fn cli_solves_3x3() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_solve"))
        .args(["3", "3", "1"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("value=Loss"), "unexpected output: {s}");
    assert!(s.contains("nodes="), "unexpected output: {s}");
    assert!(s.contains("tt_entries="), "unexpected output: {s}");
}

#[test]
fn cli_bad_args_fail() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_solve"))
        .args(["3", "3"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}
