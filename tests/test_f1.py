from src.evaluation.evaluate import token_f1


def test_token_f1_exact():
    assert token_f1("hello world", "hello world") == 1.0


def test_token_f1_partial():
    score = token_f1("docker container platform", "docker is a containerization platform")
    assert 0.0 < score < 1.0
