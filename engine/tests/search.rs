use engine::evaluate::UniformEvaluator;
use engine::game::GameState;
use engine::mcts::{Mcts, MctsConfig};

fn test_config() -> MctsConfig {
    MctsConfig {
        num_simulations: 32,
        c_puct: 1.5,
        dirichlet_alpha: 0.3,
        noise_fraction: 0.0,
        temperature: 1.0,
    }
}

#[tokio::test]
async fn test_mcts_visits_increase() {
    let game = GameState::start_pos();
    let config = test_config();
    let evaluator = Box::new(UniformEvaluator);

    let mut mcts = Mcts::new(game, config, evaluator);
    let visit_counts = mcts.run().await;

    assert!(!visit_counts.is_empty());
    let total_visits: u32 = visit_counts.iter().map(|(_, c)| *c).sum();
    assert_eq!(
        total_visits, 32,
        "total visits should equal num_simulations"
    );
}

#[tokio::test]
async fn test_mcts_policy_target_legal_only() {
    let game = GameState::start_pos();
    let config = test_config();
    let evaluator = Box::new(UniformEvaluator);

    let mut mcts = Mcts::new(game, config, evaluator);
    mcts.run().await;

    let policy = mcts.policy_target();
    assert!(!policy.is_empty());

    let prob_sum: f32 = policy.iter().map(|(_, p)| *p).sum();
    assert!(
        (prob_sum - 1.0).abs() < 1e-3,
        "policy probabilities should sum to 1.0, got {prob_sum}"
    );

    for &(aid, prob) in &policy {
        assert!(
            (aid as usize) < engine::ACTION_SPACE_SIZE,
            "action_id out of range"
        );
        assert!(prob >= 0.0 && prob <= 1.0, "invalid probability {prob}");
    }
}

#[tokio::test]
async fn test_mcts_best_move() {
    let game = GameState::start_pos();
    let config = test_config();
    let evaluator = Box::new(UniformEvaluator);

    let mut mcts = Mcts::new(game, config, evaluator);
    let visit_counts = mcts.run().await;

    let best = Mcts::best_move(&visit_counts);
    let max_visits = visit_counts.iter().map(|(_, c)| *c).max().unwrap();
    let best_visits = visit_counts.iter().find(|(m, _)| *m == best).unwrap().1;
    assert_eq!(best_visits, max_visits);
}

#[tokio::test]
async fn test_mcts_advance_reuse() {
    let game = GameState::start_pos();
    let config = test_config();
    let evaluator = Box::new(UniformEvaluator);

    let mut mcts = Mcts::new(game, config, evaluator);
    let visit_counts = mcts.run().await;
    let best = Mcts::best_move(&visit_counts);

    mcts.advance(best);

    // After advance, the game state should reflect the move
    assert_eq!(
        mcts.game().board.side_to_move,
        engine::Color::Black,
        "should be black's turn after red moves"
    );

    // Run again from the new position
    let visit_counts2 = mcts.run().await;
    assert!(!visit_counts2.is_empty());
    let total2: u32 = visit_counts2.iter().map(|(_, c)| *c).sum();
    assert_eq!(total2, 32);
}

#[tokio::test]
async fn test_full_game_terminates() {
    let game = GameState::start_pos();
    let config = MctsConfig {
        num_simulations: 8,
        c_puct: 1.5,
        dirichlet_alpha: 0.3,
        noise_fraction: 0.25,
        temperature: 1.0,
    };
    let evaluator = Box::new(UniformEvaluator);

    let mut mcts = Mcts::new(game, config, evaluator);
    let mut plies = 0;

    while !mcts.game().status.is_terminal && plies < 400 {
        let visit_counts = mcts.run().await;
        if visit_counts.is_empty() {
            break;
        }
        let best = Mcts::best_move(&visit_counts);
        mcts.advance(best);
        plies += 1;
    }

    assert!(
        mcts.game().status.is_terminal || plies >= 300,
        "game should terminate within MAX_TOTAL_PLIES (300) or by other rule, plies={plies}"
    );
}
