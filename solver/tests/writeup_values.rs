use quoridor_solver::board::Board;
use quoridor_solver::solver::{Solver, Value};

fn start_value(w: u8, h: u8, walls: u8) -> Value {
    let b = Board::new(w, h, walls);
    let s = b.initial();
    Solver::new(&b).solve(&s)
}

// ---- default set: fast, double-confirmed (smallboard + writeup) ----
#[test] fn v_3x3_w1_p2_win()  { assert_eq!(start_value(3,3,1), Value::Loss); }
#[test] fn v_4x4_w1_p1_win()  { assert_eq!(start_value(4,4,1), Value::Win);  } // even-height P1
#[test] fn v_4x4_w2_p1_win()  { assert_eq!(start_value(4,4,2), Value::Win);  }
#[test] fn v_5x5_w0_p2_win()  { assert_eq!(start_value(5,5,0), Value::Loss); }
#[test] fn v_5x5_w1_p2_win()  { assert_eq!(start_value(5,5,1), Value::Loss); }

// ---- extended set: writeup-specific; slow without Phase-1 opts; run explicitly:
//      cargo test --release -- --ignored
#[ignore] #[test] fn v_5x5_w5_p1_win() { assert_eq!(start_value(5,5,5), Value::Win);  }
#[ignore] #[test] fn v_8x3_w3_draw()   { assert_eq!(start_value(8,3,3), Value::Draw); } // GHI canary
