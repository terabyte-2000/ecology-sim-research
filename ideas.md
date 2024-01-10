# Ideas

## Original idea: 
- Simulate species with different types of cells on the grid that spread and die out by certain rules.
- Set some constraints and some parameters and run a genetic algorithm over it to see what pops out
- Analyze results (what this corresponds to in the real world) and put some graphics
- How to extend this model
- Link source code



How to simulate the multiple species at once?

- Attempt to evolve each one per its own rules; if they conflict, use a conflict management based rule instead

    * How to find conflict?
        + Have multiple cell autos that have just each species, compare them to see any overlap, resolve conflict, and render

    * Fundamentally, what do I want to do?
        + Ecological simulation with genetic algorithm (which people have already done)

    * What am I doing different than others?
        + Cellular automata? 
        + But this is a really janky way to do it that doesn't actually use cellular automata in a useful way

- Different rule based on what creature it is all incorporated into one sim
    + Better than previous b/c no conflict resolution but still not new in any meaningful way


## Pivot: Compare different methods of doing the same type of ecological simulation

- Try to recreate the same type of eco sim using neural net, convolutional NN, cellular automaton, and other computational structures
- But *what* would be compared? And *how*?
    * Performance against each other (e.g. which one has more "creatures" in the end?) 
        + Definitely would be quite strange but implementation seems reasonably straightforward
        + Simulate each one and make sure to employ the same conflict resolution as before and update the worlds and such
    * Greater diversity by some metric like diversity index, richness, etc
        + Great idea in principle but a little unsure on implementation
        + Speciation would need to somehow develop and this doesn't really work with the cellular automata as I understand it now 
    
    * Some way of measuring dynamicness or interestingness of the evolved ecology?
        + Sounds awesome but no clue how that could be measured