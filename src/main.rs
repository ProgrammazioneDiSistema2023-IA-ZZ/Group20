use petgraph::algo::toposort;

fn main() {
    println!("Hello, world!");

    use petgraph::Graph;

    let mut deps: Graph<&str, &str> = Graph::<&str, &str>::new();
    let pg = deps.add_node("petgraph");
    let fb = deps.add_node("fixedbitset");
    let qc = deps.add_node("quickcheck");
    let rand = deps.add_node("rand");
    let libc = deps.add_node("libc");
    deps.extend_with_edges(&[(pg, fb, "miao"), (pg, qc, "bau"), (qc, rand, "miao"), (rand, libc, "miao"), (qc, libc, "miao")]);

    let toposort = toposort(&deps, None).unwrap();

    for node in toposort{
        println!("{:?} - {:?}", deps[node], deps.edges_directed(node, petgraph::Direction::Incoming).map(|e| *e.weight()).collect::<Vec<&str>>());
    }

}
