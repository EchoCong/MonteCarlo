(define (domain pacman)
  (:requirements :typing :conditional-effects)
  (:types pos)				;; Coordinate of a position as one variable
  (:predicates  (At ?p - pos) 		;; Position of Pacman
                (FoodAt ?p - pos)	;; Position of food
                (CapsuleAt ?p - pos)	;; Position of capsules
                (GhostAt ?p - pos)	;; Position of Ghosts
                (Adjacent ?p1 ?p2 - pos)	;; Whether two positions are connected
                (unPowered)  ;; Whether Pacman has eaten the capsule
  )
  (:action move
        :parameters (?curPos ?nextPos - pos)
        :precondition (or (and (At ?curPos)
                            (Adjacent ?curPos ?nextPos)
                            (not (unPowered))
                       )
                        (and (At ?curPos)
                            (Adjacent ?curPos ?nextPos)
                            (unPowered)
                            (not (GhostAt ?nextPos))
                        ))
        :effect   (and 
                       (not  (At ?curPos))
                       (At ?nextPos)
                       (not  (FoodAt ?nextPos) )
                       (when (CapsuleAt ?nextPos) (not (unPowered)))
                       (not  (CapsuleAt ?nextPos) )
                       (when (not (unPowered)) (not (GhostAt ?nextPos)))
                   )
   )
)